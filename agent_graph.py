from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from memory_store import MemoryModule

class AgentState(TypedDict):
    user_query: str
    memory_context: str
    plan: str
    draft_response: str
    critique: str
    final_response: str
    iteration_count: int

def build_twin_graph(api_key):
    # 1. Initialize Memory
    memory = MemoryModule(api_key)
    
    # 2. UPDATED: Using the available 'gemini-2.5-flash' from your scan
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    def memory_node(state):
        return {"memory_context": memory.retrieve_context(state['user_query'])}

    def planner_node(state):
        prompt = f"""Goal: {state['user_query']} 
        Context: {state['memory_context']}"""
        return {"plan": llm.invoke([HumanMessage(content=prompt)]).content}

    def executor_node(state):
        prompt = f"""Plan: {state['plan']}
        Execute a detailed response."""
        return {"draft_response": llm.invoke([HumanMessage(content=prompt)]).content, 
                "iteration_count": state.get("iteration_count", 0) + 1}

    def reflector_node(state):
        prompt = f"""Critique this: {state['draft_response']}
        If perfect, say APPROVED."""
        return {"critique": llm.invoke([HumanMessage(content=prompt)]).content}

    def finalizer_node(state):
        memory.add_memory(state['user_query'], state['draft_response'])
        return {"final_response": state['draft_response']}

    workflow = StateGraph(AgentState)
    workflow.add_node("memory", memory_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("finalizer", finalizer_node)

    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")
    workflow.add_conditional_edges("reflector", 
        lambda x: "end" if "APPROVED" in x["critique"] or x["iteration_count"] >= 2 else "retry",
        {"retry": "executor", "end": "finalizer"})
    workflow.add_edge("finalizer", END)
    return workflow.compile()
