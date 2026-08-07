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
    chat_history: NotRequired[str]  # Add this!
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
    
    question = state["question"]
    context = state.get("context", "")
    draft = state.get("draft_answer", "")
    
    # 1. Define strict evaluation rules
    evaluation_prompt = f"""
    You are a strict academic reviewer grading an AI assistant's draft answer. 
    Your job is to ensure the draft accurately answers the user's question using ONLY the provided context.
    
    [USER QUESTION]
    {question}
    
    [SOURCE CONTEXT]
    {context}
    
    [DRAFT ANSWER]
    {draft}
    
    [EVALUATION RULES]
    1. Does the draft answer the question directly?
    2. Is every claim in the draft supported by the SOURCE CONTEXT? (No outside knowledge allowed).
    3. Is the draft missing any crucial details from the context that the user asked for?
    
    [OUTPUT FORMAT]
    - If the draft passes all rules, output EXACTLY the word: PASS
    - If the draft fails, output "FAILED: " followed by a brief, specific instruction on what the Retriever needs to fix.
    
    Review:
    """
    
    # 2. Call Gemini to evaluate the draft
    response = Settings.llm.complete(evaluation_prompt).text.strip()
    
    print(f"🧐 CRITIC VERDICT: {response}")
    
    # 3. Return the feedback to update the state
    return {"critic_feedback": response}

def formatter_agent(state: GraphState):
    print("✨ FORMATTER: Structuring final output with flashcards...")
    
    draft = state.get("draft_answer", "")
    question = state["question"]
    
    # 1. Define the formatting instructions
    formatting_prompt = f"""
    You are an expert educational designer. Your task is to take a raw, verified academic answer and format it into a highly readable, structured study guide.
    
    [ORIGINAL QUESTION]
    {question}
    
    [VERIFIED RAW ANSWER]
    {draft}
    
    [INSTRUCTIONS]
    1. Rewrite the raw answer into a clear, engaging explanation using Markdown formatting. Use headings, bullet points, and bold text for readability.
    2. STRICT RULE: Do NOT add new factual information. You must only structure the facts provided in the raw answer.
    3. At the end of your response, add a "## Flashcards" section. Generate 2 to 3 flashcards summarizing the core concepts from the answer. 
    
    Format the flashcards exactly like this:
    **Q:** [Question]
    **A:** [Answer]
    ---
       
    Polished Output:
    """
    
    # 2. Call Gemini to format the text
    final = Settings.llm.complete(formatting_prompt).text
    
    # 3. Update the state with the final string
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
initial_input: GraphState = {"question": "Can you tell me about Saragam aluminium company?", "loop_count": 0}
final_state = app.invoke(initial_input)

print("\n--- FINAL OUTPUT ---")
# Safely print the final answer using .get() just in case the workflow failed
print(final_state.get("final_answer", "No answer generated."))
