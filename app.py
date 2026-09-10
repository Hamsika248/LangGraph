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
# 4. TOOL - RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return standard output or error."""

    import sys
    import io
    import traceback

    if not isinstance(code, str):
        code = str(code)

    # Remove markdown code fences if Gemini returns them
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

    if result.strip():
        return result.strip()

    return "Success (no terminal output)"


# ============================================================
# 5. TOOL - GENERATE TEST CASES
# ============================================================

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

    dev_prompt = f"""
You are a Python Developer.

Write a clean Python program to solve the following coding task:

{task}

Rules:
- Return ONLY Python code.
- Do NOT include explanations.
- Do NOT use Markdown.
- The code must be executable.
"""

    response = llm_flash.invoke(dev_prompt)

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

        code_str = "\n".join(code_parts)

    else:

        code_str = str(content)

    # Remove markdown fences if present
    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code_str
    }


# ============================================================
# 7. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    task = state["messages"][-1].content

    # --------------------------------------------------------
    # Generate test scenarios
    # --------------------------------------------------------

    test_cases = generate_test_cases.invoke(
        task
    )

    cases_str = str(test_cases)

    # --------------------------------------------------------
    # Execute generated Python code
    # --------------------------------------------------------

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # --------------------------------------------------------
    # Create final report
    # --------------------------------------------------------

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
# 8. CREATE LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(CrewState)


# Add nodes
workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)


# Starting point
workflow.add_edge(
    START,
    "developer"
)


# Developer → Tester
workflow.add_edge(
    "developer",
    "tester"
)


# Tester → End
workflow.add_edge(
    "tester",
    END
)


# Compile LangGraph
agent = workflow.compile()


# ============================================================
# 9. FORMAT INPUT FOR LANGGRAPH
# ============================================================

class AgentInput(TypedDict):
    input: str


def format_for_agent(x):

    # Handle dictionary input from LangServe
    if isinstance(x, dict):
        user_input = x["input"]

    else:
        user_input = x.input

    return {
        "messages": [
            HumanMessage(
                content=user_input
            )
        ],
        "next_step": "developer",
        "code": None,
        "report": None
    }


# ============================================================
# 10. FORMAT OUTPUT FOR PLAYGROUND
# ============================================================

def extract_agent_output(state):

    if not isinstance(state, dict):

        return {
            "generated_code": str(state),
            "report": ""
        }

    generated_code = state.get(
        "code",
        ""
    )

    report = state.get(
        "report",
        ""
    )

    return {
        "generated_code": generated_code or "",
        "report": report or ""
    }


# ============================================================
# 11. CREATE LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_agent_output)
)


# ============================================================
# 12. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph Coding Agent",
    description=(
        "A LangGraph Developer and Tester "
        "Coding Agent"
    ),
    version="1.0"
)


# ============================================================
# 13. EXPOSE AGENT THROUGH LANGSERVE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 14. ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph Coding Agent is running",
        "status": "active"
    }


# ============================================================
# 15. START SERVER
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
