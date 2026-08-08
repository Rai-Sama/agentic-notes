import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv() 

import chromadb
from google.genai.errors import APIError as GoogleAPIError
from groq import APIError as GroqAPIError
from llama_cloud import LlamaCloud
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

# ==========================================
# 1. FLEET CONFIGURATION
# ==========================================
print("Configuring Fleet...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

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

def call_llm_with_fallback(prompt: str) -> str:
    global current_llm_index
    page_success = False
    result_text = ""
    
    while current_llm_index < len(llm_fleet) and not page_success:
        Settings.llm = llm_fleet[current_llm_index]
        for attempt in range(5):
            try:
                result_text = Settings.llm.complete(prompt).text
                time.sleep(4) # Rate limit protection
                page_success = True
                break 
            # FIX: Catch all exceptions so no underlying client library can bypass the fallback
            except Exception as e:
                print(f"⚠️ API Error on LLM #{current_llm_index + 1}: {e}")
                print("Pausing 25s before retry...")
                time.sleep(25)
        
        if not page_success:
            print(f"❌ LLM #{current_llm_index + 1} completely failed. Rotating...")
            current_llm_index += 1

    if not page_success:
        raise RuntimeError("FATAL: Exhausted all LLMs.")
        
    return result_text

# ==========================================
# 2. CHROMA DB & NODE PARSER SETUP
# ==========================================
print("Connecting to ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("student_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize the index so we can insert into it dynamically
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=1, 
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# ==========================================
# 3. BATCH FOLDER INGESTION (FILE-BY-FILE)
# ==========================================
client = LlamaCloud()
notes_dir = Path("./student_notes")
processed_dir = Path("./processed_notes")
processed_dir.mkdir(exist_ok=True)

valid_extensions = {".pdf", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"}

files_to_process = [f for f in notes_dir.iterdir() if f.suffix.lower() in valid_extensions]

if not files_to_process:
    print("✅ No new files found in ./student_notes. Everything is up to date!")
    sys.exit(0)

for file_path in files_to_process:
    print(f"\n--- Processing: {file_path.name} ---")
    
    # 1. PARSE
    file_upload = client.files.create(file=file_path, purpose="parse")
    result = client.parsing.parse(
        tier="agentic", 
        version="latest",
        file_id=file_upload.id,
        expand=["markdown"],
        agentic_options={
            "custom_prompt": """
            This document contains handwritten notes, diagrams, and text with non-linear spatial layouts.
            CRITICAL INSTRUCTIONS:
            1. Follow the visual flow of arrows and spatial grouping.
            2. DIAGRAMS & IMAGES: Output a placeholder EXACTLY like this: [DIAGRAM: descriptive_name.png]
            3. Provide a highly detailed textual description of what the diagram shows immediately after.
            """
        }
    )
    
    # 2. CLEAN
    valid_pages = []
    if result.markdown and result.markdown.pages:
        for i, page in enumerate(result.markdown.pages):
            page_text = getattr(page, "markdown", None)
            if isinstance(page_text, str) and page_text.strip():
                print(f"Cleaning {file_path.name} - page {i + 1}...")
                cleaning_prompt = f"""
                You are a data engineer cleaning raw OCR text from student notes.
                CRITICAL RULES:
                1. Maintain original Markdown formatting.
                2. Do NOT summarize or add commentary.
                3. PRESERVE ALL [DIAGRAM: ...] placeholders and their descriptions exactly.
                
                [RAW OCR TEXT]
                {page_text}
                
                [CLEANED TEXT]
                """
                valid_pages.append(call_llm_with_fallback(cleaning_prompt))
                
    full_text = "\n\n".join(valid_pages)
    
    if not full_text.strip():
        print(f"⚠️ Warning: No valid text extracted from {file_path.name}. Skipping.")
        continue

    output_path = f"./parsed_{file_path.stem}_debug.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"💾 Saved debug markdown to {output_path}")

    # 3. CHUNK AND EMBED (Atomic Commit)
    print(f"Embedding {file_path.name} into ChromaDB...")
    
    if not full_text.strip():
        print(f"⚠️ Warning: No valid text extracted from {file_path.name}. Skipping.")
        continue

    # 3. CHUNK AND EMBED (Atomic Commit)
    print(f"Embedding {file_path.name} into ChromaDB...")
    doc = Document(text=full_text, metadata={"file_name": file_path.name})
    nodes = node_parser.get_nodes_from_documents([doc])
    
    # Insert directly into the live index
    index.insert_nodes(nodes)
    
    # 4. ARCHIVE (File successfully processed and embedded!)
    destination = processed_dir / file_path.name
    shutil.move(str(file_path), str(destination))
    print(f"✅ Success! Moved {file_path.name} to processed_notes/")

print("\n🎉 All incremental ingestion complete! Ready for graph.py")
