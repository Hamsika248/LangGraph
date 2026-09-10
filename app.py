
import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from typing import TypedDict, List, Optional


# ============================================================
# 1. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


# ============================================================
# 2. INITIALIZE GEMINI MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

llm = llm_flash


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):
    messages: List
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 4. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the output or error."""

    import sys
    import io
    import traceback

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 test scenarios for a coding task."""

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for the following coding task:

{task_description}

Include:
1. Standard test cases
2. Edge cases
3. Boundary cases where applicable

Return them as a numbered list.
"""

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# ============================================================
# 5. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    task = state["messages"][-1].content

    dev_prompt = f"""
You are a Python Developer.

Write a clean Python script to solve this coding task:

{task}

Rules:
- Return ONLY Python code
- Do not provide explanations
- Do not use Markdown
- Make the code executable
"""

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        if isinstance(content[0], dict):
            code_str = content[0].get("text", "")
        else:
            code_str = str(content[0])

    else:
        code_str = str(content)

    return {
        "code": code_str
    }


# ============================================================
# 6. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    task = state["messages"][-1].content

    # Generate test cases
    test_cases = generate_test_cases.invoke(task)

    if isinstance(test_cases, list):

        if isinstance(test_cases[0], dict):
            cases_str = test_cases[0].get("text", "")
        else:
            cases_str = str(test_cases[0])

    else:
        cases_str = str(test_cases)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # Create report
    report = f"""
### EXECUTION OUTPUT

{execution_result}


### TEST SCENARIOS EVALUATED

{cases_str}
"""

    return {
        "report": report
    }


# ============================================================
# 7. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    END
)

agent = workflow.compile()


# ============================================================
# 8. INPUT FORMAT
# ============================================================

class AgentInput(TypedDict):
    input: str


def format_for_agent(x):

    user_input = x["input"]

    return {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "next_step": None,
        "code": None,
        "report": None
    }


# ============================================================
# 9. OUTPUT FORMAT
# ============================================================

def extract_agent_output(state):

    return {
        "generated_code": state.get("code", ""),
        "report": state.get("report", "")
    }


# ============================================================
# 10. CREATE API CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_agent_output)
).with_types(input_type=AgentInput)


# ============================================================
# 11. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph Coding Agent",
    description="Developer and Tester Coding Agent"
)


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 12. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
