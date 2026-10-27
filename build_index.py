"""Build the Chroma index in chroma_db/ from the PDFs in documents/.

Run this again whenever the PDFs change, then commit chroma_db/.
Embeds in small batches with pauses to stay under the free tier's
100 embedding requests per minute.
"""

import os
import shutil
import time

import tomllib
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if "GOOGLE_API_KEY" not in os.environ:
    with open(os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"), "rb") as f:
        os.environ["GOOGLE_API_KEY"] = tomllib.load(f)["GOOGLE_API_KEY"]

# Import after the key is set; app.py's Streamlit calls are harmless outside `streamlit run`
from app import EMBEDDING_MODEL, INDEX_DIRECTORY, load_chunks  # noqa: E402

BATCH_SIZE = 25

chunks = load_chunks()
shutil.rmtree(INDEX_DIRECTORY, ignore_errors=True)
vectorstore = Chroma(
    collection_name="Company_documents",
    embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=INDEX_DIRECTORY,
)
for start in range(0, len(chunks), BATCH_SIZE):
    if start:
        time.sleep(61)
    for attempt in range(5):
        try:
            vectorstore.add_documents(chunks[start:start + BATCH_SIZE])
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            print("Rate limited, waiting a minute...")
            time.sleep(61)
    print(f"Embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
print("Index saved to", INDEX_DIRECTORY)
