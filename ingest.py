import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() 

import chromadb
import nest_asyncio
from google.genai.errors import APIError as GoogleAPIError
from groq import APIError as GroqAPIError
from llama_cloud import LlamaCloud
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

nest_asyncio.apply()

print("Configuring free models and Fallback Fleet...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- 1. BUILD THE LLM FLEET ---
llm_fleet = []

if os.getenv("GROQ_API_KEY"):
    llm_fleet.append(Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY")))
if os.getenv("GOOGLE_API_KEY_1"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY_1")))
if os.getenv("GOOGLE_API_KEY_2"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY_2")))
if os.getenv("GOOGLE_API_KEY_3"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY_3")))
if os.getenv("GOOGLE_API_KEY_1"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY_1")))
if os.getenv("GOOGLE_API_KEY_2"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY_2")))
if os.getenv("GOOGLE_API_KEY_3"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY_3")))
if not llm_fleet:
    raise ValueError("No API keys found to build the LLM Fleet!")

current_llm_index = 0
Settings.llm = llm_fleet[current_llm_index]
print(f"Loaded {len(llm_fleet)} models in the fallback cascade.")
# ------------------------------

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
    expand=["markdown"],
    agentic_options={
        "custom_prompt": """
        This document contains handwritten notes with non-linear spatial layouts, mind-maps, and arrows. 
        CRITICAL INSTRUCTIONS:
        1. DO NOT simply read top-to-bottom left-to-right. 
        2. Follow the visual flow of arrows and spatial grouping. 
        3. If a concept points to sub-items (e.g., "3 major forces" pointing to other words), you MUST group those items together under a Markdown heading or bulleted list.
        4. Keep distinct case studies and examples separated from the main theoretical points.
        """
    }
)

print("Cleaning OCR and formatting with Gemini (ELT Step)...")

# Initialize an empty string
full_text = ""

# 1. Safely check that the markdown object actually exists to satisfy Pyright
if result.markdown and result.markdown.pages:
    
    valid_pages = []
    # 2. Iterate safely since we proved it's not None
    for i, page in enumerate(result.markdown.pages):
        page_text = getattr(page, "markdown", None)
        
        if isinstance(page_text, str) and page_text.strip():
            print(f"Cleaning page {i + 1}...")
            
            cleaning_prompt = f"""
            You are an expert data engineer cleaning raw OCR text from student notes.
            Your task is to fix typos, correct broken sentences, and fix garbled reading orders.
            
            CRITICAL RULES:
            1. Maintain the original Markdown formatting (headers, bullet points, bold text).
            2. DO NOT add any outside information, commentary, or conversational text.
            3. DO NOT summarize. Keep the full length of the notes.
            4. If the text is already clean, return it exactly as is.
            
            [RAW OCR TEXT]
            {page_text}
            
            [CLEANED TEXT]
            """
            
            # --- FALLBACK & RETRY LOGIC ---
            page_success = False
            
            # Outer Loop: Try the current LLM, rotate if it completely fails
            while current_llm_index < len(llm_fleet) and not page_success:
                Settings.llm = llm_fleet[current_llm_index]
                max_retries = 5
                
                # Inner Loop: Attempt the request up to max_retries times
                for attempt in range(max_retries):
                    try:
                        # 1. Attempt the API call
                        cleaned_text = Settings.llm.complete(cleaning_prompt).text
                        valid_pages.append(cleaned_text)
                        
                        # 2. Success! Mark the flag, sleep briefly, and break the inner loop
                        print(f"Success on LLM #{current_llm_index + 1}! Sleeping 4s...")
                        time.sleep(4)
                        page_success = True
                        break 
                        
                    except (GoogleAPIError, GroqAPIError) as e:
                        # 3. An error occurred (Rate limit, quota, server down, etc.)
                        wait_time = 25
                        print(f"⚠️ Error on LLM #{current_llm_index + 1}: {e.__class__.__name__}. Pausing {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                
                # 4. If we finished the inner loop and still don't have success, rotate the key
                if not page_success:
                    print(f"❌ LLM #{current_llm_index + 1} completely failed after {max_retries} attempts. Rotating to next fallback...")
                    current_llm_index += 1

            # 5. If we exhausted all keys and retries, crash gracefully
            if not page_success:
                raise RuntimeError(f"FATAL: Exhausted all LLM fallbacks and retries on page {i + 1}.")
            # ------------------------------
    # Join the fully cleaned and verified pages
    full_text = "\n\n".join(valid_pages)

    if not full_text:
        raise RuntimeError("Parsing finished, but no valid markdown text was returned.")

    # --- NEW: SAVE THE MARKDOWN LOCALLY ---
    output_path = "./parsed_notes_debug.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"💾 Saved parsed markdown to '{output_path}' for review.")
    # --------------------------------------

    documents = [Document(text=full_text, metadata={"file_name": "student_notes.pdf"})]
    print("Successfully extracted and cleaned markdown!")

if not full_text:
    raise RuntimeError("Parsing finished, but no valid markdown text was returned.")

documents = [Document(text=full_text, metadata={"file_name": "marketing_101.pdf"})]
print("Successfully extracted and cleaned markdown!")

from llama_index.core.node_parser import SentenceWindowNodeParser

print("Chunking document using Sentence Windows...")

# Initialize the parser. window_size=2 means each sentence gets 
# the 2 sentences before it and the 2 sentences after it attached as metadata.
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Extract the nodes
nodes = node_parser.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} sentence nodes!")

print("Connecting to ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("student_notes")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Embedding and saving to ChromaDB...")
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context
)

query_engine = index.as_query_engine()

print("Querying...")
response = query_engine.query("What are the main topics covered in these notes?")

print("\n--- RAG Response ---")
print(str(response))
