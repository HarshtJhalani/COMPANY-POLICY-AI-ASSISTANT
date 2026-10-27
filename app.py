"""Streamlit web demo of the Company Policy AI Assistant (Streamlit Community Cloud).

Same pipeline as company_policy_ai_assistant.ipynb, but reads the PDFs from
the local `documents/` folder, the Gemini key from the GOOGLE_API_KEY secret,
and uses Gemini embeddings instead of a local model to stay within free-tier memory.
"""

import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), "documents")

st.set_page_config(page_title="Company Policy AI Assistant", page_icon="📄")

# On Streamlit Cloud the key comes from the app's Secrets; locally, from the environment
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Missing GOOGLE_API_KEY. Add it to the app's Secrets.")
        st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)


INDEX_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_chunks():
    pdf_docs = []
    for name in sorted(os.listdir(PDF_DIRECTORY)):
        if not name.lower().endswith(".pdf"):
            continue
        docs = PyPDFLoader(os.path.join(PDF_DIRECTORY, name)).load()
        for doc in docs:
            doc.metadata["document_name"] = name
        pdf_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
    return chunks


@st.cache_resource(show_spinner="Loading the policy index...")
def build_index():
    """Load the prebuilt index (made by build_index.py) once per server.

    Embedding all chunks at startup would exceed the free tier's 100 embedding
    requests per minute, so the index is built offline and committed.
    """
    vectorstore = Chroma(
        collection_name="Company_documents",
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=INDEX_DIRECTORY,
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
    document_names = sorted({m["document_name"] for m in metadatas})
    return retriever, document_names


retriever, DOCUMENT_NAMES = build_index()


def format_documents(documents, label, max_chars=None):
    parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:max_chars] if max_chars else doc.page_content
        parts.append(
            f"{label} {i}\n"
            f"Document: {doc.metadata.get('document_name', 'Unknown Document')}\n"
            f"Page: {doc.metadata.get('page', 0) + 1}\n\n"
            f"{content}\n"
        )
    return "\n".join(parts)


RAG_PROMPT = """
You are the Company Policy Answering Engine.

Answer the employee's question ONLY using the policy
information provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent company policies.
3. Do not assume information that is not explicitly present.
4. If the provided information does not answer the question,
   respond exactly that the information could not be found
   in the provided company policy documents.
5. If multiple policies are relevant, combine them carefully.
6. Mention important conditions, eligibility requirements,
   exceptions, or limitations.
7. Cite the document and page number used.
8. Keep the answer clear and concise.

POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

ANSWER:
"""


def text_of(content):
    """Gemini may return a plain string or a list of content parts."""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


@tool
def company_policy_rag(question: str) -> str:
    """
    Search company policy documents and answer an employee's
    question using only the retrieved policy information.
    Use this tool when the employee needs a factual answer
    from company policy documents.
    """
    context = format_documents(retriever.invoke(question), "SOURCE")
    response = llm.invoke(RAG_PROMPT.format(context=context, question=question))
    return text_of(response.content)


@tool
def search_policy_documents(query: str) -> str:
    """
    Search the company policy documents and return relevant
    document excerpts with their document names and page numbers.
    Use this when evidence or source information needs to be inspected.
    """
    documents = retriever.invoke(query)
    if not documents:
        return "No relevant policy documents were found."
    return format_documents(documents, "Result", max_chars=800)


@tool
def calculate_policy_value(expression: str) -> str:
    """
    Perform basic arithmetic calculations needed when answering
    policy-related numerical questions.

    Example:
    '12 - 5'
    '20 * 0.5'
    """
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "Invalid mathematical expression."
    try:
        return f"Calculation result: {eval(expression, {'__builtins__': {}}, {})}"
    except Exception:
        return "Unable to perform calculation."


@tool
def get_remaining_holidays(start_date: str = None, end_date: str = "2026-12-31") -> str:
    """
    Retrieve the company holiday policy/document and identify
    holidays occurring between the specified dates.

    The start_date and end_date can be in any readable format.
    If start_date is not specified or is empty, today's date will be used.
    """
    try:
        start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    except Exception:
        start_date_str = datetime.now().strftime("%Y-%m-%d")
    if not start_date or not str(start_date).strip():
        start_date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    except Exception:
        end_date_str = "2026-12-31"

    documents = retriever.invoke(
        "Data Axle 2026 Indian holidays public holidays national holidays holiday dates"
    )
    if not documents:
        return "No holiday information was found in the company documents."

    prompt = f"""
You are extracting holiday information from company documents.

Use ONLY the information provided below.

Find all company/public/national holidays that fall
between {start_date_str} and {end_date_str}, inclusive.

Return:
1. Holiday name
2. Date
3. Day of week
4. Source document
5. Page number

Do not invent or add holidays that are not present
in the provided documents.

If the documents do not contain enough information,
say so clearly.

DOCUMENTS:

{format_documents(documents, "Document")}
"""
    return text_of(llm.invoke(prompt).content)


AGENT_INSTRUCTIONS = """
You are an AI-powered Company Policy Assistant.

Your job is to help employees understand information
contained in the provided company policy documents.

You have access to four tools:

1. company_policy_rag
   Use this to answer factual questions using the
   company policy documents.

2. search_policy_documents
   Use this to inspect policy evidence, source documents,
   page numbers, or relevant excerpts.

3. calculate_policy_value
   Use this only for arithmetic calculations that are
   based on information retrieved from the policies.

4. get_remaining_holidays
   Use this when the employee asks about:
   - remaining national or public holidays
   - holidays between two dates
   - holidays remaining until a specific date
   - the number of remaining holidays
   - upcoming holidays listed in the company holiday calendar

   This tool must obtain holiday dates from the
   company holiday calendar document.
   Never use hard-coded holiday dates or outside knowledge.

IMPORTANT RULES:

- Company policy documents are the ONLY source of truth.
- Never use outside knowledge for company policy questions.
- Never invent a policy.
- Never assume an eligibility condition.
- When information cannot be found, clearly say so.
- For holiday-related questions, use the get_remaining_holidays tool
  instead of relying on general knowledge about holidays.
- Today's date is {today}.
- For numerical questions involving policy information, retrieve the
  relevant policy information before performing calculations.
- Use calculate_policy_value only after the relevant policy information
  has been retrieved.
- When multiple policies are relevant, consider all of them.
- Give concise answers.
- Mention the relevant policy document and page whenever possible.
- If the requested information is not present in the company documents,
  clearly state that it could not be found in the provided company
  policy documents.
- Do not fabricate dates, numbers, eligibility conditions,
  exceptions, or policy rules.
"""

agent = create_agent(
    model=llm,
    tools=[company_policy_rag, search_policy_documents, calculate_policy_value, get_remaining_holidays],
)


def respond(message):
    try:
        response = agent.invoke({
            "messages": [
                {"role": "system", "content": AGENT_INSTRUCTIONS.format(today=datetime.now().strftime("%Y-%m-%d"))},
                {"role": "user", "content": message},
            ]
        })
        return text_of(response["messages"][-1].content)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "The free Gemini quota is used up for now. Please wait a minute and try again."
        return f"Sorry, something went wrong: {e}"


EXAMPLES = [
    "How many leaves in a year? Give sick and casual separately.",
    "Where is the WFH policy mentioned?",
    "How many public holidays are remaining until 31 Dec 2026?",
    "Does the company provide free gym memberships?",
]

st.title("📄 Company Policy AI Assistant")
st.caption(
    "Ask about leave, HR, WFH and holiday policies. Answers come only from the "
    "policy documents and cite document and page. Demo with made-up documents."
)

with st.sidebar:
    st.subheader("Documents")
    for name in DOCUMENT_NAMES:
        st.markdown(f"- {name}")
    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a policy question") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the policies..."):
            answer = respond(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
