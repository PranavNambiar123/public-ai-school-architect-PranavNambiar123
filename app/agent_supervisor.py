# Step 1: Import necessary modules and classes
# Fill in any additional imports you might need
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import operator
import requests
import asyncio
import aiohttp

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

@tool
def make_post_request(url: str, data: dict) -> str:
    """Make a POST request to the specified URL with the given data."""
    try:
        response = requests.post(url, json=data)
        return f"Status: {response.status_code}, Response: {response.text}"
    except Exception as e:
        return f"Error making request: {str(e)}"

@tool
async def stress_test_endpoint(url: str, iterations: int = 10) -> str:
    """Make multiple concurrent POST requests to test endpoint stability."""
    async def single_request(session, url):
        try:
            async with session.post(url) as response:
                return response.status == 200
        except:
            return False

    async with aiohttp.ClientSession() as session:
        tasks = [single_request(session, url) for _ in range(iterations)]
        results = await asyncio.gather(*tasks)
        success_rate = sum(results) / len(results)
        return f"Success rate: {success_rate * 100}%, {sum(results)}/{len(results)} requests succeeded"

# Step 3: Define the system prompt for the supervisor agent
# Customize the members list as needed.
members = ["Researcher", "Coder", "Reviewer", "Tester", "StressTester"]

system_prompt = f"""
You are the supervisor of a team of {', '.join(members)}.
You are responsible for coordinating the team to complete tasks efficiently.
You have the following members: {', '.join(members)}.
Each worker will perform their assigned task and provide results.
You will analyze their output and decide the next step or finish when the task is complete.
When all required work is done, respond with FINISH.
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
                  f"Or should we FINISH? Select one of {', '.join(options)}")
    ]
).partial(options=str(options), members=', '.join(members))

# Step 6: Define the prompt for the supervisor agent
# Customize the prompt if needed.

# Step 7: Initialize the language model
# Choose the model you need, e.g., "gpt-4o"
llm = ChatOpenAI(model="gpt-4")

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
    final_output: Optional[str] = None

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

coder_agent = create_agent(
    llm,
    [python_repl_tool],
    """You are an expert Python developer specializing in FastAPI endpoints.
    Your main task is to create and implement POST endpoints with proper request/response models.
    Always include proper error handling and input validation.
    Use FastAPI best practices and type hints."""
)
coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

reviewer_agent = create_agent(llm, [tavily_tool], "You are a senior developer. You excel at code review. You provide detailed and actionable feedback.")
reviewer_node = functools.partial(agent_node, agent=reviewer_agent, name="Reviewer")

tester_agent = create_agent(llm, [python_repl_tool], "You are a safe developer who generates test cases using unittest.")
tester_node = functools.partial(agent_node, agent=tester_agent, name="Tester")

stress_tester_agent = create_agent(
    llm,
    [make_post_request, stress_test_endpoint],
    """You are a performance testing specialist.
    Your role is to stress test POST endpoints by making multiple concurrent requests.
    Always perform 10 concurrent requests to test endpoint stability.
    Report success rates and any failures encountered."""
)
stress_tester_node = functools.partial(agent_node, agent=stress_tester_agent, name="StressTester")

# Step 13: Define the workflow using StateGraph
# Add nodes and their corresponding functions to the workflow.
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Reviewer", reviewer_node)
workflow.add_node("Tester", tester_node)
workflow.add_node("StressTester", stress_tester_node)

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
