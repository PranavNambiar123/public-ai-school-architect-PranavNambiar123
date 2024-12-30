# Step 1: Import necessary modules and classes
# Fill in any additional imports you might need
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import operator

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END

# Step 2: Define tools
# Here, define any tools the agents might use. Example given:
tavily_tool = TavilySearchResults(max_results=5)

# This tool executes code locally, which can be unsafe. Use with caution:
python_repl_tool = PythonREPLTool()

# Step 3: Define the system prompt for the supervisor agent
# Customize the members list as needed.
members = ["Researcher", "Coder", "Reviewer", "QA Tester"]

system_prompt = f"""
You are the supervisor of a team of {', '.join(members)}.
You are responsible for ensuring that the team members perform their tasks in a timely and efficient manner.
You have the following members: {', '.join(members)}.
You need to coordinate the team members to ensure that they perform their tasks in a timely and efficient manner.
Each worker will perform the task and respond with their results. When finished respond with FINISH.
"""

# Step 4: Define options for the supervisor to choose from
options = members + ["FINISH"]

# Step 5: Define the function for OpenAI function calling
# Define what the function should do and its parameters.
function_def = {
    "name": "route",
    "description": "Route the task to the appropriate worker",
    "parameters": {
        "title": "Route the task to the appropriate worker",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Given the conversation above who should act next?\n"
                  f"Or should we FINISH? Select one of {', '.join(options)}"),
        ("human", "{input}"),
    ]
).partial(options=str(options), members=', '.join(members))

# Step 6: Define the prompt for the supervisor agent
# Customize the prompt if needed.

# Step 7: Initialize the language model
# Choose the model you need, e.g., "gpt-4o"
llm = ChatOpenAI(model="gpt-4o")

# Step 8: Create the supervisor chain
# Define how the supervisor chain will process messages.
supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# Step 9: Define a typed dictionary for agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Step 10: Function to create an agent
# Fill in the system prompt and tools for each agent you need to create.
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Step 11: Function to create an agent node
# This function processes the state through the agent and returns the result.
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Step 12: Create agents and their corresponding nodes
# Define the specific role and tools for each agent.
researcher_agent = create_agent(llm, [tavily_tool], "You are a web researcher")
researcher_node = functools.partial(agent_node, agent=researcher_agent, name="Researcher")

coder_agent = create_agent(llm, [python_repl_tool], "You generate safe code to analyze data with pandas and generate charts using matplotlib")
coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

reviewer_agent = create_agent(llm, [tavily_tool], "You are a senior developer. You excel at code review. You provide detailed and actionable feedback.")
reviewer_node = functools.partial(agent_node, agent=reviewer_agent, name="Reviewer")

tester_agent = create_agent(llm, [python_repl_tool], "You are safe developer who generates sage classes using unittest.")
tester_node = functools.partial(agent_node, agent=tester_agent, name="Tester")

# Step 13: Define the workflow using StateGraph
# Add nodes and their corresponding functions to the workflow.
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Reviewer", reviewer_node)
workflow.add_node("QA Tester", tester_node)

# Step 14: Add edges to the workflow
# Ensure that all workers report back to the supervisor.
for member in members:
    workflow.add_edge(member, "supervisor")

# Step 15: Define conditional edges
# The supervisor determines the next step or finishes the process.
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Step 16: Set the entry point
workflow.set_entry_point("supervisor")

# Step 17: Compile the workflow into a graph
# This creates the executable workflow.
graph = workflow.compile()
