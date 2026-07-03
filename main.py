import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage,AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
st.title("Story Generator")
llm_model = init_chat_model(model="gemini-2.5-flash-lite",model_provider="google_genai")

class State(TypedDict):
    query: str 
    story : str 
    title : str 

def story_agent(state: State):
    prompt = f'''you are an assistant who will take query {state["query"]} from user and build story on that'''
    response = llm_model.invoke(prompt)
    story = response.content
    return {"story": story}

story_graph = StateGraph(State)
story_graph.add_node("story_agent",story_agent)
story_graph.add_edge(START,"story_agent")
story_graph.add_edge("story_agent",END)

agent_story = story_graph.compile()


def title_agent(state: State):
    # story = state["story"].content
    prompt = f'''based on the story {state["story"]} you have to suggest title'''
    response = llm_model.invoke(prompt)
    title = response.content
    return {"title": title}

title_graph = StateGraph(State)
title_graph.add_node("title_agent",title_agent)
title_graph.add_edge(START, "title_agent")
title_graph.add_edge("title_agent", END)

agent_title = title_graph.compile()

def orchestration(state: State):
    story = agent_story.invoke(state)
    title = agent_title.invoke(story)
    return {"story": story["story"], "title": title["title"]}

orchestration_graph = StateGraph(State)
orchestration_graph.add_node("orchestration",orchestration)
orchestration_graph.add_edge(START,"orchestration")
orchestration_graph.add_edge("orchestration",END)

agent_final = orchestration_graph.compile()

userinput = st.text_input("Enter your query...")

if st.button("send"):
    query = {"query": userinput}
    if query is not None: 
        ans = agent_final.invoke(query)
        st.write(
            ans["query"], 
            "\n\n***********************************************************************\n\n",
            ans["title"], 
            "\n\n***********************************************************************\n\n",
            ans["story"]
            )
