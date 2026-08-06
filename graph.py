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
    
    # 1. Use ChromaDB query_engine here to fetch context
    # context = query_engine.query(state["question"]).source_nodes
    
    # 2. Use Gemini to write a draft answer based on context + feedback
    # If the critic sent feedback from a previous loop, include it in the prompt!
    draft = f"Draft answer for: {state['question']}"
    
    # Increment the loop count so we don't get stuck forever
    current_loops = state.get("loop_count", 0) + 1
    
    return {"draft_answer": draft, "loop_count": current_loops, "context": "..."}

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
