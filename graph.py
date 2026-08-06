import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# 1. Load keys and configure models
load_dotenv()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = GoogleGenAI(model="gemini-3.5-flash-lite")

# 2. Connect to the existing ChromaDB
print("Connecting to ChromaDB...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("student_notes")

# 3. Rebuild the index from the local database
from llama_index.vector_stores.chroma import ChromaVectorStore

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# This loads the index without needing to re-embed the PDF!
index = VectorStoreIndex.from_vector_store(
    vector_store, 
    storage_context=storage_context
)

# 4. Create a Retriever (fetches text, but doesn't auto-generate answers)
retriever = index.as_retriever(similarity_top_k=3)

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired


class GraphState(TypedDict):
    question: str
    loop_count: int
    context: NotRequired[str]
    draft_answer: NotRequired[str]
    critic_feedback: NotRequired[str]
    final_answer: NotRequired[str]

def retriever_agent(state: GraphState):
    print("🕵️ RETRIEVER: Fetching context and writing draft...")
    question = state["question"]
    feedback = state.get("critic_feedback", "")
    
    # 1. Fetch the most relevant chunks from ChromaDB
    nodes = retriever.retrieve(question)
    
    # Combine the text from the retrieved chunks
    context = "\n\n".join([node.get_content() for node in nodes])
    
    # 2. Build the prompt dynamically
    prompt = (
        "You are a helpful study assistant. Answer the question using ONLY the provided context.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
    )
    
    # 3. If the graph looped back, inject the Critic's instructions!
    if feedback:
        print(f"🕵️ RETRIEVER: Adjusting based on feedback: {feedback}")
        prompt += f"PREVIOUS ATTEMPT FEEDBACK:\nA reviewer found this issue with your last draft: '{feedback}'. Please fix it in this new draft.\n\n"
        
    prompt += "Draft Answer:"
    
    # 4. Generate the draft using the globally configured Gemini model
    draft = Settings.llm.complete(prompt).text
    
    # Increment the loop count
    current_loops = state.get("loop_count", 0) + 1
    
    return {"draft_answer": draft, "loop_count": current_loops, "context": context}

def critic_agent(state: GraphState):
    print("🧐 CRITIC: Reviewing the draft against the context...")
    
    # Use Gemini to compare state["draft_answer"] against state["context"]
    # Ask Gemini to output either "PASS" or a list of corrections.
    feedback = "FAILED: Missing details about X" # Or "PASS"
    
    return {"critic_feedback": feedback}

def formatter_agent(state: GraphState):
    print("✨ FORMATTER: Structuring final output with flashcards...")
    
    # Use Gemini to turn state["draft_answer"] into Markdown/Flashcards
    final = "Here is your beautiful answer with flashcards!"
    
    return {"final_answer": final}

def routing_decision(state: GraphState):
    # Safely fetch the feedback, defaulting to an empty string if missing
    feedback = state.get("critic_feedback", "")
    
    # If Gemini outputted "PASS", go to the Formatter
    if "PASS" in feedback:
        print("➡️ ROUTER: Draft passed! Sending to Formatter.")
        return "formatter"
    
    # If it failed, check the loop count. 
    # If we already looped, accept it as-is and force it to the Formatter.
    if state["loop_count"] >= 2:
        print("➡️ ROUTER: Max loops reached. Forcing to Formatter.")
        return "formatter"
    
    # Otherwise, loop back to the Retriever for a rewrite!
    print("➡️ ROUTER: Issues found. Looping back to Retriever.")
    return "retriever"

# Initialize the graph with our state definition
workflow = StateGraph(GraphState)

# 1. Add our Agent nodes
workflow.add_node("retriever", retriever_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("formatter", formatter_agent)

# 2. Define the strict flow
workflow.add_edge(START, "retriever") # Always start here
workflow.add_edge("retriever", "critic") # Retriever always hands off to Critic
workflow.add_edge("formatter", END) # Formatter is always the last step

# 3. Add the conditional loop from Critic -> Formatter OR Retriever
workflow.add_conditional_edges(
    "critic", 
    routing_decision,
    # Map the strings returned by routing_decision to actual node names
    {
        "formatter": "formatter",
        "retriever": "retriever"
    }
)

# 4. Compile it!
app = workflow.compile()

# --- TEST THE LOOP ---
print("\n--- STARTING WORKFLOW ---")

# Explicitly type-hint the dictionary as GraphState
initial_input: GraphState = {"question": "What is the capital of France?", "loop_count": 0}
final_state = app.invoke(initial_input)

print("\n--- FINAL OUTPUT ---")
# Safely print the final answer using .get() just in case the workflow failed
print(final_state.get("final_answer", "No answer generated."))
