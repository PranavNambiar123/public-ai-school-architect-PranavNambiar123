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

@tool
def analyze_code_style(code: str) -> str:
    """Analyze code for PEP 8 compliance and common Python style issues."""
    try:
        # Simulate style checking
        issues = []
        if "    " in code:  # Check for spaces instead of tabs
            issues.append("- Use spaces for indentation")
        if len(max(code.split('\n'), key=len)) > 79:  # Check line length
            issues.append("- Line too long (>79 characters)")
        return "\n".join(issues) if issues else "Code follows PEP 8 style guide"
    except Exception as e:
        return f"Error analyzing code: {str(e)}"

# Step 3: Define the system prompt for the supervisor agent
# Customize the members list as needed.
members = [
    "Researcher", "Coder", "Reviewer", "Tester", "DocumentationWriter", "Linter",
    "SecurityAuditor", "Optimizer", "DesignPatternExpert"
]

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
researcher_agent = create_agent(
    llm, 
    [tavily_tool], 
    "You are a web researcher who excels at finding relevant information and best practices."
)
researcher_node = functools.partial(agent_node, agent=researcher_agent, name="Researcher")

coder_agent = create_agent(
    llm,
    [python_repl_tool],
    """You are an expert Python developer.
    Focus on writing clean, maintainable code following Python best practices.
    Always include proper error handling and type hints.
    Break down complex functionality into smaller, reusable components."""
)
coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

reviewer_agent = create_agent(
    llm, 
    [tavily_tool], 
    """You are a senior developer who excels at code review.
    Focus on code quality, architecture, and potential improvements.
    Provide detailed and actionable feedback."""
)
reviewer_node = functools.partial(agent_node, agent=reviewer_agent, name="Reviewer")

tester_agent = create_agent(
    llm, 
    [python_repl_tool], 
    """You are a test engineer who creates comprehensive test cases.
    Focus on unit tests, edge cases, and test coverage.
    Use unittest and provide clear test documentation."""
)
tester_node = functools.partial(agent_node, agent=tester_agent, name="Tester")

documentation_agent = create_agent(
    llm,
    [python_repl_tool],
    """You are a technical documentation specialist.
    Create clear, comprehensive documentation following Google style guide.
    Include:
    - Detailed function/class documentation
    - Usage examples
    - Installation instructions
    - API documentation
    - README files
    Focus on making the documentation accessible to both new and experienced developers."""
)
documentation_node = functools.partial(agent_node, agent=documentation_agent, name="DocumentationWriter")

linter_agent = create_agent(
    llm,
    [analyze_code_style],
    """You are a code quality specialist focusing on Python style and formatting.
    Your responsibilities:
    - Ensure PEP 8 compliance
    - Check code formatting
    - Identify potential code smells
    - Suggest improvements for code readability
    - Verify consistent coding style
    Always provide specific, actionable feedback for improvements."""
)
linter_node = functools.partial(agent_node, agent=linter_agent, name="Linter")

security_agent = create_agent(
    llm,
    [python_repl_tool],
    """You are a security audit specialist focusing on Python applications.
    Your responsibilities:
    - Identify potential security vulnerabilities
    - Review code for common security issues (OWASP Top 10)
    - Suggest secure coding practices
    - Review dependency security
    - Recommend security improvements
    Always provide specific examples and secure alternatives."""
)
security_node = functools.partial(agent_node, agent=security_agent, name="SecurityAuditor")

optimizer_agent = create_agent(
    llm,
    [python_repl_tool],
    """You are a performance optimization expert.
    Your responsibilities:
    - Identify performance bottlenecks
    - Suggest algorithmic improvements
    - Optimize memory usage
    - Improve code efficiency
    - Recommend caching strategies
    Focus on measurable performance improvements while maintaining code readability."""
)
optimizer_node = functools.partial(agent_node, agent=optimizer_agent, name="Optimizer")

design_pattern_agent = create_agent(
    llm,
    [python_repl_tool],
    """You are a design pattern and architecture expert.
    Your responsibilities:
    - Recommend appropriate design patterns
    - Identify architectural improvements
    - Suggest SOLID principle applications
    - Review code structure and organization
    - Propose refactoring for better design
    Focus on maintainable and scalable architecture solutions."""
)
design_pattern_node = functools.partial(agent_node, agent=design_pattern_agent, name="DesignPatternExpert")

# Step 13: Define the workflow using StateGraph
# Add nodes and their corresponding functions to the workflow.
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Reviewer", reviewer_node)
workflow.add_node("Tester", tester_node)
workflow.add_node("DocumentationWriter", documentation_node)
workflow.add_node("Linter", linter_node)
workflow.add_node("SecurityAuditor", security_node)
workflow.add_node("Optimizer", optimizer_node)
workflow.add_node("DesignPatternExpert", design_pattern_node)

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
