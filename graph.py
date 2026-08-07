import os
import time
from typing import TypedDict

import chromadb
from dotenv import load_dotenv
from google.genai.errors import APIError as GoogleAPIError
from groq import APIError as GroqAPIError
from langgraph.graph import END, START, StateGraph
from llama_index.core import Settings, StorageContext, VectorStoreIndex
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
if os.getenv("GOOGLE_API_KEY_1"):
    llm_fleet.append(GoogleGenAI(model="models/gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY_1")))

if not llm_fleet:
    raise ValueError("No API keys found to build the LLM Fleet!")

current_llm_index = 0

def call_llm_with_fallback(prompt: str) -> str:
    """Helper function to run any prompt through the fault-tolerant fleet."""
    global current_llm_index
    page_success = False
    result_text = ""
    
    while current_llm_index < len(llm_fleet) and not page_success:
        Settings.llm = llm_fleet[current_llm_index]
        max_retries = 3 # Shorter retries for interactive chat
        
        for attempt in range(max_retries):
            try:
                result_text = Settings.llm.complete(prompt).text
                page_success = True
                break 
            except (GoogleAPIError, GroqAPIError):
                wait_time = 5 
                print(f"\n⚠️ API Error on LLM #{current_llm_index + 1}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        if not page_success:
            print(f"\n❌ LLM #{current_llm_index + 1} completely failed. Rotating to next fallback...")
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

def retriever_agent(state: GraphState):
    print("🕵️ RETRIEVER: Fetching context and writing draft...")
    question = state["question"]
    history = state.get("chat_history", "")
    feedback = state.get("critic_feedback", "")
    loop_count = state.get("loop_count", 0)
    
    if feedback and loop_count > 0:
        refine_prompt = (
            f"The previous search for '{question}' failed because: '{feedback}'\n"
            "Generate a new search query using different keywords from the feedback to find the correct section in the notes.\n"
            "STRICT RULE: Output ONLY the raw search query string. No conversational filler.\n"
            "New Search Query:"
        )
        search_query = call_llm_with_fallback(refine_prompt).strip()
        print(f"🔄 CRAG RE-RETRIEVAL QUERY: {search_query}")
    else:
        search_query = question

    nodes = retriever.retrieve(search_query)
    sources = list({node.metadata.get("file_name", "student_notes.pdf") for node in nodes})
    sources_str = ", ".join(sources) if sources else "Uploaded Notes"

    context = "\n\n".join([node.get_content() for node in nodes])
    
    # --- TEMPORARY DEBUG PRINT ---
    print("\n" + "="*50)
    print("🔍 DEBUG - WHAT CHROMADB RETURNED:")
    print(context[:800] + "...\n[Context truncated for display]")
    print("="*50 + "\n")
    # -----------------------------
    
    prompt = f"""
    You are an academic extractor. Answer the user's question using ONLY the provided context.
    
    CRITICAL RULES:
    1. BLIND OBEDIENCE: If the context says the sky is green, you say the sky is green. Do not use outside knowledge. 
    2. IGNORE EXAMPLES: If the context contains specific industry case studies (like Skincare or Food Delivery), DO NOT include them in your answer unless the user specifically asked for examples. Extract only the general theory.
    
    Question: {question}
    
    Context:
    {context}
    """
    
    if feedback:
        prompt += f"\nCRITIC FEEDBACK TO ADDRESS: {feedback}\n"
    
    prompt += "\nDraft Answer:"
    draft = call_llm_with_fallback(prompt)
    
    return {
        "draft_answer": draft, 
        "loop_count": loop_count + 1, 
        "context": context,
        "sources": sources_str
    }

def critic_agent(state: GraphState):
    print("🧐 CRITIC: Evaluating draft for logical coherence...")
    question = state["question"]
    context = state.get("context", "")
    draft = state.get("draft_answer", "")
    
    evaluation_prompt = f"""
    You are an evaluator checking a draft against the SOURCE CONTEXT.
    
    [USER QUESTION]
    {question}
    
    [SOURCE CONTEXT]
    {context}
    
    [DRAFT ANSWER]
    {draft}
    
    [CRITICAL EVALUATION RULES]
    1. PARAMETRIC BLINDNESS: You MUST NOT judge the draft based on standard economic theory (like "supply and demand"). You must ONLY judge it against the SOURCE CONTEXT. If the context gives a weird, brief, or non-standard definition, and the draft accurately reports it, you MUST PASS IT.
    2. COHERENCE: Did the draft successfully avoid mixing up specific industry case studies (like skincare) with the general theory?
    
    [OUTPUT FORMAT]
    - If the draft accurately reflects the source context (even if the context is brief), output EXACTLY the word: PASS
    - If it hallucinated outside knowledge or mixed up case studies, output "FAILED: " followed by a brief instruction.
    
    Review:
    """
    
    response = call_llm_with_fallback(evaluation_prompt).strip()
    print(f"🧐 CRITIC VERDICT: {response}")
    return {"critic_feedback": response}
def fallback_agent(state: GraphState):
    """Triggered when notes fail to provide a complete answer after retries."""
    print("🤖 FALLBACK: Context in notes was insufficient. Using Frontier Knowledge...")
    question = state["question"]
    
    fallback_prompt = (
        "You are an expert academic tutor. The user asked a question that was not fully covered in their class notes.\n"
        "Provide a complete, accurate, and highly educational answer using your general knowledge.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    
    # Call your highest-reasoning frontier model (Groq / Gemini 3.6 Flash)
    frontier_answer = call_llm_with_fallback(fallback_prompt)
    
    disclaimer = (
        "\n\n> ⚠️ **Note:** *Your uploaded notes did not contain complete details on this topic. "
        "This response was generated using general academic knowledge.*"
    )
    
    return {"final_answer": frontier_answer + disclaimer}

def formatter_agent(state: GraphState):
    print("✨ FORMATTER: Structuring final output with flashcards & citations...")
    draft = state.get("draft_answer", "")
    question = state["question"]
    sources = state.get("sources", "Uploaded Notes")
    
    formatting_prompt = f"""
    You are an expert educational designer. Format this raw answer into a structured study guide.
    
    [ORIGINAL QUESTION]
    {question}
    
    [VERIFIED RAW ANSWER]
    {draft}
    
    [INSTRUCTIONS]
    1. Rewrite into a clean, structured explanation using Markdown headers and bullet points.
    2. Do NOT add new factual information.
    3. At the end, add a "## Flashcards" section (2-3 flashcards).
    
    Polished Output:
    """
    
    final = call_llm_with_fallback(formatting_prompt)
    
    # Append the source citation at the very bottom
    final_with_sources = f"{final}\n\n---\n**📚 Sources Referenced:** `{sources}`"
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

workflow = StateGraph(GraphState)

# 1. ADD THE FALLBACK NODE
workflow.add_node("retriever", retriever_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("formatter", formatter_agent)
workflow.add_node("fallback", fallback_agent) 

# 2. DEFINE THE STANDARD EDGES
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "critic")
workflow.add_edge("formatter", END)
workflow.add_edge("fallback", END) # Fallback also goes directly to END

# 3. UPDATE THE CONDITIONAL ROUTING MAP
workflow.add_conditional_edges(
    "critic", 
    routing_decision, 
    {
        "formatter": "formatter", 
        "retriever": "retriever",
        "fallback": "fallback" # The router is now allowed to use this path!
    }
)

app = workflow.compile()

# ==========================================
# 4. START CONTINUOUS CHAT
# ==========================================
print("\n🚀 Multi-Agent Study System Initialized! (Type 'exit' to quit)")

running_history = ""
# Explicitly type-hint the dictionary as GraphState
initial_input: GraphState = {"question": "what does saragam aluminium company manufacture?", "loop_count": 0}
final_state = app.invoke(initial_input)
running_history = ""

while True:
    user_input = input("\n📝 You: ")
    
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye! Happy studying.")
        break

    initial_input: GraphState = {
        "question": user_input,
        "loop_count": 0,
        "chat_history": running_history
    }
    
    final_state = app.invoke(initial_input)
    answer = final_state.get("final_answer", "Error: No answer generated.")
    print(f"\n✨ System:\n{answer}")
    
    # Store history for conversational context
    running_history += f"User: {user_input}\nSystem: {answer}\n\n"
