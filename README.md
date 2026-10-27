# Company Policy AI Assistant

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://company-policy-assistant-v1.streamlit.app/)

**Live demo:** https://company-policy-assistant-v1.streamlit.app/ (uses made-up policy documents)

A RAG-based AI agent that answers employee questions from company policy PDFs. Answers cite the source document and page, and the assistant says so when the answer is not in the documents instead of guessing.

## What it does
- Loads policy PDFs from Google Drive and splits them into chunks
- Retrieves the most relevant chunks for a question and answers only from them
- Cites the document name and page number in every answer
- Refuses out-of-scope questions (for example "What is Formula 1?" returns "information could not be found in the provided company policy documents")
- Uses an agent that picks the right tool for each question

## Tech stack
- **LLM:** Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`, temperature 0.2
- **Framework:** LangChain (agent with tool calling)
- **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** ChromaDB (top-5 similarity retrieval)
- **Document loading:** PyPDFLoader, RecursiveCharacterTextSplitter (chunk size 500, overlap 50)
- **Environment:** Google Colab

## How it works
1. **Ingest:** 4 PDFs (HR policies, company policies, leave policies, 2026 holiday calendar) are loaded, 15 pages in total, and split into 114 chunks. Each chunk keeps its document name and page number.
2. **Index:** chunks are embedded and stored in ChromaDB.
3. **Answer:** a grounded prompt tells the model to use only the retrieved text, not invent policies, state conditions and exceptions, and cite sources.
4. **Agent:** the agent chooses between four tools:
   - `company_policy_rag`: answers a policy question from retrieved context
   - `search_policy_documents`: returns raw excerpts with document and page, for checking evidence
   - `calculate_policy_value`: arithmetic on retrieved numbers (input restricted to digits and operators)
   - `get_remaining_holidays`: reads the holiday PDF and lists holidays between two dates

## Example questions
- "How many leaves in a year? Give sick and casual separately." (leave policy: Earned 18, Casual 8, Sick 10)
- "Where is the WFH policy mentioned?"
- "How many public holidays are remaining until 31 Dec 2026?"
- "Does the company provide free gym memberships?" (not in the documents, so the assistant says so)

## Setup
1. Open `company_policy_ai_assistant.ipynb` in Google Colab.
2. Put your policy PDFs in a Drive folder and set `pdf_directory` to its path.
3. Add your Gemini API key as a Colab secret and read it with `userdata.get(...)`.
4. Run all cells, then ask questions with `ask_policy("your question")`.

## Evaluation
Tested on a hand-written set of 35 questions with known answers ([eval/questions.csv](eval/questions.csv)): 29 answerable from the documents (facts, multi-part, calculations, holiday dates) and 6 out of scope.

| Metric | Result |
|---|---|
| Retrieval hit@5 (right page among the 5 retrieved chunks) | 29/29 (100%) |
| Answer accuracy (expected values present, no refusal) | 29/29 (100%) |
| Citation accuracy (right document and page cited) | 29/29 (100%) |
| Out-of-scope questions refused | 6/6 (100%) |
| Wrong refusals of answerable questions | 0 |

Answers are graded with keyword and regex checks rather than an LLM judge, to keep cost low. Per-question answers are in [eval/results.csv](eval/results.csv). Run it yourself with `python eval/eval.py` (needs `GOOGLE_API_KEY`).

Caveats: this is a small set over 15 pages of clean, made-up documents, from a single run, so it shows the pipeline works as designed rather than how it would do on large or messy real-world documents. In the first run the grader itself made two mistakes (it took "not covered" as a refusal and missed a "Pages 1–2" citation); both checks were fixed and the saved answers re-scored without calling the model again.

## Limitations and next steps
- Plain similarity search; no re-ranking or hybrid search yet
- No conversation memory yet: each question is answered independently
- Evaluation is a small keyword-graded set; a larger set and RAGAS-style faithfulness scoring would be the next step
- Document upload: let visitors try their own PDF (one file, about 10 pages) in a private index for their session, alongside the built-in documents. The holiday tool would be turned off for uploaded documents, since it expects the company holiday calendar

## Web demo
`app.py` is a Streamlit chat app for Streamlit Community Cloud. It uses Gemini embeddings (`gemini-embedding-001`) instead of the local MiniLM model so it fits the free tier's memory.
- Put the policy PDFs in `documents/`.
- Locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, add your key, then `pip install -r requirements.txt` and `streamlit run app.py`.
- On Streamlit Cloud: add `GOOGLE_API_KEY` in the app's Secrets.

## License
MIT
