import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel

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
# 2. INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):
    messages: List
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 4. RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the output."""

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

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():
        return result.strip()

    return "Success (no terminal output)"


# ============================================================
# 5. GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test scenarios for a coding task."""

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 test scenarios for this coding task:

{task_description}

Include:
1. Standard test cases
2. Edge cases
3. Boundary cases where applicable

Return the test scenarios as a numbered list.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):
                text_parts.append(
                    item.get("text", "")
                )
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# 6. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    task = state["messages"][-1].content

    prompt = f"""
You are a Python Developer.

Write a clean Python program to solve this coding task:

{task}

Rules:
- Return ONLY Python code.
- Do NOT include explanations.
- Do NOT use Markdown.
- The code must be executable.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:

            if isinstance(item, dict):
                code_parts.append(
                    item.get("text", "")
                )
            else:
                code_parts.append(str(item))

        code = "\n".join(code_parts)

    else:

        code = str(content)

    # Remove Markdown code fences
    code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code
    }


# ============================================================
# 7. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    task = state["messages"][-1].content

    # Generate test cases
    test_cases = generate_test_cases.invoke(task)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    report = f"""
### EXECUTION OUTPUT

{execution_result}


### TEST SCENARIOS

{test_cases}
"""

    return {
        "report": report
    }


# ============================================================
# 8. CREATE LANGGRAPH
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
# 9. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str


# ============================================================
# 10. RUN THE GRAPH
# ============================================================

def run_agent(x):

    # Get user's input
    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    # Create initial LangGraph state
    initial_state = {
        "messages": [
            HumanMessage(
                content=user_input
            )
        ],
        "next_step": "developer",
        "code": None,
        "report": None
    }

    # IMPORTANT:
    # Actually invoke the LangGraph
    result = agent.invoke(initial_state)

    return result


# ============================================================
# 11. FORMAT OUTPUT
# ============================================================

def extract_agent_output(state):

    return {
        "generated_code": state.get("code", "") or "",
        "report": state.get("report", "") or ""
    }


# ============================================================
# 12. CREATE LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(run_agent)
    | RunnableLambda(extract_agent_output)
).with_types(
    input_type=AgentInput
)


# ============================================================
# 13. FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph Coding Agent",
    description="LangGraph Developer and Tester Coding Agent",
    version="1.0"
)


# ============================================================
# 14. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 15. ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Coding Agent is running",
        "status": "active"
    }


# ============================================================
# 16. START SERVER
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
