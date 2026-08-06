import os
from pathlib import Path

# 1. NEW: Import and load the .env file securely
from dotenv import load_dotenv

load_dotenv() 

import chromadb
import nest_asyncio
from llama_cloud import LlamaCloud
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

nest_asyncio.apply()

# 2. NEW: Fail fast if the .env file is missing or empty
if not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Missing API keys! Make sure your .env file is set up.")

print("Configuring free models...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = GoogleGenAI(model="models/gemini-3.5-flash-lite")

# The client automatically looks for the LLAMA_CLOUD_API_KEY in the environment
client = LlamaCloud()

print("Uploading and parsing PDF via LlamaCloud...")
file_upload = client.files.create(
    file=Path("./student_notes.pdf"),
    purpose="parse"
)

# Use .parse() instead of .create(). This blocks and waits automatically!
print("Parsing document (this will wait automatically until finished)...")
result = client.parsing.parse(
    tier="agentic", 
    version="latest",
    file_id=file_upload.id,
    expand=["markdown"] 
)

# The API returns markdown per page. We join it together into one big string.
# Initialize an empty string
full_text = ""

# 1. Safely check that the markdown object actually exists
if result.markdown and result.markdown.pages:
    
    # 2. Iterate through the pages and only grab the text if the page didn't fail
    valid_pages = []
    for page in result.markdown.pages:
        # getattr fetches the 'markdown' property, but returns None if it doesn't exist.
        # This completely avoids the direct `page.markdown` dot-access that Pyright hates.
        page_text = getattr(page, "markdown", None)
        
        # Check if we got a valid string back
        if isinstance(page_text, str) and page_text.strip():
            valid_pages.append(page_text)
            
    full_text = "\n\n".join(valid_pages)
if not full_text:
    raise RuntimeError("Parsing finished, but no valid markdown text was returned.")

documents = [Document(text=full_text)]
print("Successfully extracted into markdown!")

print("Connecting to ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("student_notes")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Embedding and saving to ChromaDB...")
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

query_engine = index.as_query_engine()

print("Querying...")
response = query_engine.query("What are the main topics covered in these notes?")

print("\n--- RAG Response ---")
print(str(response))
