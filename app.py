import asyncio
import os
import shutil
from pathlib import Path
from typing import TypedDict

import chainlit as cl
import chromadb
from dotenv import load_dotenv
from google.genai.errors import APIError as GoogleAPIError
from groq import APIError as GroqAPIError
from langgraph.graph import END, START, StateGraph
from llama_cloud import LlamaCloud
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing_extensions import NotRequired

load_dotenv()

# ==========================================
# 1. FLEET & FALLBACK CONFIGURATION
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
if os.getenv("GOOGLE_API_KEY_1"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY_1")))

if not llm_fleet:
    raise ValueError("No API keys found to build the LLM Fleet!")

current_llm_index = 0

# 1. MADE THIS ASYNC
async def call_llm_with_fallback(prompt: str) -> str:
    global current_llm_index
    page_success = False
    result_text = ""
    
    while current_llm_index < len(llm_fleet) and not page_success:
        Settings.llm = llm_fleet[current_llm_index]
        max_retries = 3 
        
        for attempt in range(max_retries):
            try:
                # Use await and acomplete() instead of complete()
                response = await Settings.llm.acomplete(prompt)
                result_text = response.text
                page_success = True
                break 
            except (GoogleAPIError, GroqAPIError):
                wait_time = 5 
                await asyncio.sleep(wait_time) # Use asyncio.sleep instead of time.sleep
        
        if not page_success:
            current_llm_index += 1

    if not page_success:
        raise RuntimeError("FATAL: Exhausted all LLM fallbacks during chat.")
        
    return result_text

# ==========================================
# 2. DATABASE & RETRIEVER CONFIGURATION
# ==========================================
print("Connecting to ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("student_notes")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# BUGFIX: Added the postprocessor to re-expand the sentence windows!
postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
retriever = index.as_retriever(
    similarity_top_k=3, 
    node_postprocessors=[postprocessor]
)

# Configure parsing tools for file uploads
client = LlamaCloud()
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=1, 
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Ensure the processed notes directory exists
processed_dir = Path("./processed_notes")
processed_dir.mkdir(exist_ok=True)

# ==========================================
# 3. LANGGRAPH STATE & AGENTS
# ==========================================
class GraphState(TypedDict):
    question: str
    loop_count: int
    chat_history: NotRequired[str]
    context: NotRequired[str]
    draft_answer: NotRequired[str]
    critic_feedback: NotRequired[str]
    final_answer: NotRequired[str]
    sources: NotRequired[str]

async def retriever_agent(state: GraphState):
    async with cl.Step(name="🕵️ Retriever Agent") as step:
        question = state["question"]
        history = state.get("chat_history", "")
        feedback = state.get("critic_feedback", "")
        loop_count = state.get("loop_count", 0)
        
        # SCENARIO 1: The Critic rejected the last draft. 
        # We must refine the search based on the feedback.
        if feedback and loop_count > 0:
            step.input = f"Refining search based on critic feedback: {feedback}"
            refine_prompt = (
                f"The previous search for '{question}' failed because: '{feedback}'\n"
                "Generate a new search query using different keywords from the feedback to find the correct section in the notes.\n"
                "STRICT RULE: Output ONLY the raw search query string. No conversational filler.\n"
                "New Search Query:"
            )
            search_query = (await call_llm_with_fallback(refine_prompt)).strip()
            
        # SCENARIO 2: First attempt at a question, AND we have past chat history.
        # We must rewrite the query in case it contains pronouns like "it" or "they".
        elif history:
            step.input = "Contextualizing follow-up question..."
            rewrite_prompt = (
                "Given the following conversation history, rewrite the user's new question into a standalone search query. "
                "If it is a completely new topic, just return the original question.\n\n"
                f"History:\n{history}\n\n"
                f"New Question: {question}\n\n"
                "Standalone Query:"
            )
            search_query = (await call_llm_with_fallback(rewrite_prompt)).strip()
            
        # SCENARIO 3: Brand new session, no history, no feedback.
        else:
            search_query = question

        step.input = f"Searching notes for: {search_query}"
        
        nodes = retriever.retrieve(search_query)
        sources = list({node.metadata.get("file_name", "student_notes.pdf") for node in nodes})
        sources_str = ", ".join(sources) if sources else "Uploaded Notes"

        # Inject the file name directly above each chunk of text
        context_chunks = []
        for node in nodes:
            file_name = node.metadata.get("file_name", "Unknown File")
            text = node.get_content()
            context_chunks.append(f"--- SOURCE: {file_name} ---\n{text}")
            
        context = "\n\n".join(context_chunks)

        prompt = f"""
        You are an academic extractor. Answer the user's question using ONLY the provided context.
        CRITICAL RULES:
        1. BLIND OBEDIENCE: If the context says the sky is green, you say the sky is green.
        2. IGNORE EXAMPLES: Do not include specific case studies unless asked.
        3. INLINE CITATIONS: Every factual claim you extract MUST end with an inline citation indicating which file it came from based on the SOURCE tags provided. Use the exact format: [File: filename.pdf].
        
        Question: {question}
        
        Context:
        {context}
        """

        if feedback:
            prompt += f"\nCRITIC FEEDBACK TO ADDRESS: {feedback}\n"
        prompt += "\nDraft Answer:"
        
        draft = await call_llm_with_fallback(prompt)
        
        # Output the draft to the UI so the user can read the Retriever's raw work!
        step.output = draft 
        
        return {
            "draft_answer": draft, 
            "loop_count": loop_count + 1, 
            "context": context,
            "sources": sources_str
        }


async def critic_agent(state: GraphState):
    async with cl.Step(name="🧐 Critic Agent") as step:
        question = state["question"]
        context = state.get("context", "")
        draft = state.get("draft_answer", "")
        
        step.input = "Evaluating draft against uploaded notes..."
        
        evaluation_prompt = f"""
        You are an evaluator checking a draft against the SOURCE CONTEXT.
        [USER QUESTION] {question}
        [SOURCE CONTEXT] {context}
        [DRAFT ANSWER] {draft}
        
        [CRITICAL EVALUATION RULES]
        1. PARAMETRIC BLINDNESS: Judge ONLY against the SOURCE CONTEXT.
        2. COHERENCE: Did the draft successfully avoid mixing up case studies?
        
        [OUTPUT FORMAT]
        - PASS
        - FAILED: [Instruction]
        """
        
        response = (await call_llm_with_fallback(evaluation_prompt)).strip()
        
        # Display the Critic's verdict in the UI
        step.output = response 
        
        # Give the step a red failure icon if it rejected the draft
        if "FAILED" in response:
            step.is_error = True 
            
        return {"critic_feedback": response}


async def fallback_agent(state: GraphState):
    async with cl.Step(name="🤖 Fallback Agent (Frontier Knowledge)") as step:
        step.input = "Notes lacked sufficient context. Activating general knowledge fallback..."
        question = state["question"]
        
        fallback_prompt = (
            "You are an expert academic tutor. The user asked a question that was not fully covered in their class notes.\n"
            "Provide a complete, accurate, and highly educational answer using your general knowledge.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        
        frontier_answer = await call_llm_with_fallback(fallback_prompt)
        disclaimer = "\n\n> ⚠️ **Note:** *Your uploaded notes did not contain complete details on this topic. This response was generated using general academic knowledge.*"
        
        final = frontier_answer + disclaimer
        step.output = "Fallback response generated."
        return {"final_answer": final}


async def formatter_agent(state: GraphState):
    async with cl.Step(name="✨ Formatter Agent") as step:
        step.input = "Formatting validated draft into rich study guide and interactive flashcards..."
        draft = state.get("draft_answer", "")
        question = state["question"]
        sources = state.get("sources", "Uploaded Notes")
        
        formatting_prompt = f"""
        You are an expert educational designer. Format this raw answer into a rich, structured study guide.
        
        [ORIGINAL QUESTION] {question}
        [VERIFIED RAW ANSWER] {draft}
        
        [FORMATTING INSTRUCTIONS]
        1. STRUCTURING: Write a clean explanation using Markdown headers (##) and bullet points.
        2. CLEAN CITATIONS (HOVER TOOLTIPS): The raw answer contains inline citations like [File: name.pdf]. To prevent visual clutter, you MUST convert EVERY one of these into a clean HTML tooltip. 
           - Use this exact syntax: <span title="name.pdf" style="cursor: help; color: #3498db; font-weight: bold;"><sup>[i]</sup></span>
           - Replace "name.pdf" with the actual file name.
           - DO NOT output the raw text "[File: name.pdf]" anywhere in the final response.
        3. MATH & EQUATIONS: You MUST use LaTeX for any formulas or single-letter variables. 
           - For block equations, put $$ on their own separate lines.
           - For inline variables, carefully enclose them with $ (e.g., $P$, $H$, $S$).
           - 🚨 CRITICAL: Do NOT leave unclosed $ signs (like SH$ or $PS). This will break the UI renderer completely!
           - Do NOT put spaces between the $ and the formula (e.g., $O(N)$ or $$\\mu = \\frac{{\\sum x_i}}{{N}}$$).
        4. COLOR CODING: CONSISTENTLY highlight the core concept in EVERY bullet point or section. Ensure uniform styling across the entire output using this exact syntax: <span style="color: #e67e22; font-weight: bold;">Your Keyword</span>.
        5. MEMORY TRICK: Create a short, clever mnemonic, analogy, or memory trick to help retain the core concept. Format it as a blockquote starting with: "> 🧠 **Memory Trick:**"
        6. INTERACTIVE FLASHCARDS: At the very end, add a "## Flashcards" section. Generate 2-3 flashcards using HTML <details> and <summary> tags. 
        🚨 STRICT NEGATIVE CONSTRAINT: DO NOT use Markdown links or anchor tags (like [click to reveal](#)). You MUST output the raw HTML exactly as formatted below.
        
        Use EXACTLY this flashcard HTML structure and nothing else:
        <details>
        <summary>🎯 <b>Q: [Write the question here]</b></summary>
        <div style="padding: 10px; margin-top: 5px; border-left: 3px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1);">
        <b>A:</b> [Write the answer here]
        </div>
        </details>
        <br>
        
        Polished Output:
        """

        final = await call_llm_with_fallback(formatting_prompt)
        final_with_sources = f"{final}\n\n---\n**📚 Sources Referenced:** `{sources}`"
        
        step.output = "Rich formatting complete!"
        return {"final_answer": final_with_sources}

def routing_decision(state: GraphState):
    feedback = state.get("critic_feedback", "")
    
    # 1. Pass -> Go to Formatter
    if "PASS" in feedback:
        print("➡️ ROUTER: Draft passed! Sending to Formatter.")
        return "formatter"
    
    # 2. Failed twice -> Fall back to Frontier Model Knowledge
    if state["loop_count"] >= 2:
        print("➡️ ROUTER: Notes lack necessary context after retries. Routing to Frontier Fallback!")
        return "fallback"
    
    # 3. Failed once -> Try Re-Retrieval with dynamic query adjustment
    print("➡️ ROUTER: Critic identified missing info. Re-retrieving with refined query...")
    return "retriever"

async def ingest_file_async(file_path: Path):
    async with cl.Step(name="📄 Ingesting Notes") as step:
        step.input = f"Processing {file_path.name}..."
        
        # 1. PARSE with LlamaCloud
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
                    step.output = f"Cleaning page {i + 1}..."
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
                    valid_pages.append(await call_llm_with_fallback(cleaning_prompt))
                    
        full_text = "\n\n".join(valid_pages)
        
        if not full_text.strip():
            step.is_error = True
            step.output = f"⚠️ Warning: No valid text extracted from {file_path.name}."
            return False

        # 3. CHUNK AND EMBED
        doc = Document(text=full_text, metadata={"file_name": file_path.name})
        nodes = node_parser.get_nodes_from_documents([doc])
        index.insert_nodes(nodes)
        
        # 4. ARCHIVE
        destination = processed_dir / file_path.name
        shutil.move(str(file_path), str(destination))
        
        step.output = f"✅ Success! {file_path.name} is now in your study database."
        return True

# Build the workflow globally
workflow = StateGraph(GraphState)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("formatter", formatter_agent)
workflow.add_node("fallback", fallback_agent) 

workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "critic")
workflow.add_edge("formatter", END)
workflow.add_edge("fallback", END) 
workflow.add_conditional_edges(
    "critic", 
    routing_decision, 
    {"formatter": "formatter", "retriever": "retriever", "fallback": "fallback"}
)

# Compile the async app
app = workflow.compile()

@cl.on_chat_start
async def on_chat_start():
    # Set up the chat history memory for this specific user session
    cl.user_session.set("chat_history", "")
    await cl.Message(content="🚀 **Multi-Agent Study System Initialized!**\n\nAsk me anything about your uploaded notes.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # 1. Handle File Uploads First
    if message.elements:
        for element in message.elements:
            # Type safety: Ensure the element actually has a path and mime type
            if element.path and element.mime:
                mime_type = element.mime.lower()
                
                if "pdf" in mime_type or "image" in mime_type or "powerpoint" in mime_type or "presentation" in mime_type:
                    temp_path = Path(element.path)
                    await ingest_file_async(temp_path)
        
        # If the user only uploaded a file and didn't type a question, stop here.
        if not message.content.strip():
            await cl.Message(content="✅ Files processed and embedded successfully! What would you like to know about them?").send()
            return
            
    # 2. Proceed with normal Agent workflow
    running_history = cl.user_session.get("chat_history") or ""
    
    initial_input: GraphState = {
        "question": message.content,
        "loop_count": 0,
        "chat_history": running_history
    }
    
    final_state = await app.ainvoke(initial_input)
    answer = final_state.get("final_answer", "Error: No answer generated.")
    
    await cl.Message(content=answer).send()
    
    running_history += f"User: {message.content}\nSystem: {answer}\n\n"
    cl.user_session.set("chat_history", running_history)
