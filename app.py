"""
app.py
------
UrbanPulse: an AI-powered Decision Intelligence assistant for smart urban
mobility, built for the "AI for Better Living and Smarter Communities"
challenge.

Gemini 2.5 Flash (via Vertex AI) autonomously calls predictive-analytics
tools defined in tools.py -- traffic congestion forecasting, transit
ridership demand forecasting, parking availability prediction, and mobility
anomaly detection -- and weaves the results into natural-language answers
and recommendations for citizens, commuters, and city planners. Any tool
result containing an hourly time series is automatically rendered as a chart
in the chat, so the predictive model's output is always visible, not just
described.

Run locally:
    export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    export GOOGLE_CLOUD_LOCATION="us-central1"
    streamlit run app.py --server.port 8080 --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from tools import AVAILABLE_TOOLS, TOOL_DISPATCH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_AGENT_TURNS = 5  # Safety cap on the tool-calling reasoning loop.

SYSTEM_INSTRUCTION = (
    "You are UrbanPulse, an AI Decision Intelligence assistant for smart "
    "urban mobility, built for city planners, transit operators, and "
    "everyday commuters. You have access to predictive analytics tools for "
    "traffic congestion forecasting, public transit ridership demand, "
    "parking availability prediction, and mobility anomaly detection. Use "
    "these tools whenever they can give a more accurate, data-grounded "
    "answer than your own knowledge. After using a tool, explain the "
    "forecast in plain language, mention the confidence level where "
    "relevant, and offer a concrete, actionable recommendation (e.g. best "
    "time to travel, whether to add transit capacity, where to expect "
    "delays). If a tool call fails or returns no data, say so honestly "
    "instead of guessing."
)

st.set_page_config(
    page_title="UrbanPulse | Smart Mobility Decision Intelligence",
    page_icon="🚦",
    layout="centered",
)

# Keys inside a tool result that indicate an hourly time series suitable for
# automatic charting, mapped to a human-readable y-axis label.
CHARTABLE_SERIES = {
    "congestion_percent": "Congestion (%)",
    "passengers_per_hour": "Passengers / hour",
    "availability_percent": "Available parking (%)",
}


# ---------------------------------------------------------------------------
# Client initialization (cached across reruns within a session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_genai_client() -> genai.Client:
    """Initializes and caches the google-genai Client configured for
    Vertex AI. Cached with st.cache_resource so the client (and its
    underlying connection pool) is reused across Streamlit reruns instead
    of being recreated on every user interaction.
    """
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )


def build_tool_config() -> types.GenerateContentConfig:
    """Builds the GenerateContentConfig, wiring in the system instruction
    and the Python tool functions from tools.py. The google-genai SDK
    automatically converts plain, type-hinted, docstring-documented Python
    functions into Gemini FunctionDeclarations.
    """
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=AVAILABLE_TOOLS,
        # We manage the function-calling loop manually (see run_agent_turn)
        # so we can surface each tool call and its forecast chart in the UI.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        temperature=0.4,
    )


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initializes conversational memory and UI-facing chat history in
    Streamlit's session_state on first load."""
    if "chat_history" not in st.session_state:
        # `chat_history` holds google-genai `types.Content` objects and is
        # the true conversational memory sent back to the model every turn.
        st.session_state.chat_history: List[types.Content] = []

    if "display_messages" not in st.session_state:
        # `display_messages` holds simplified dicts used purely to render
        # the chat bubbles in the Streamlit UI (including tool-call traces).
        st.session_state.display_messages: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Charting helper
# ---------------------------------------------------------------------------
def render_forecast_chart(tool_result: Dict[str, Any]) -> None:
    """Detects an hourly time series inside a tool result (e.g.
    'hourly_series') and renders it as a Streamlit line chart, so predictive
    forecasts are always shown visually alongside the numeric answer."""
    series = tool_result.get("hourly_series")
    if not series or not isinstance(series, list):
        return

    try:
        df = pd.DataFrame(series).set_index("hour")
        # Rename the single data column to something readable, if recognized.
        for col in df.columns:
            if col in CHARTABLE_SERIES:
                df = df.rename(columns={col: CHARTABLE_SERIES[col]})
        st.line_chart(df, height=220)
    except Exception:
        # Charting is a nice-to-have; never let it break the main response.
        pass


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def execute_tool_call(function_call: types.FunctionCall) -> Dict[str, Any]:
    """Executes a single tool call requested by Gemini and returns its
    result as a plain dictionary. Any exception raised by the underlying
    tool is caught and returned as an error payload so the model can react
    gracefully instead of the whole agent loop crashing.
    """
    tool_name = function_call.name
    tool_args = dict(function_call.args or {})

    if tool_name not in TOOL_DISPATCH:
        return {"error": f"Unknown tool '{tool_name}' was requested."}

    try:
        result = TOOL_DISPATCH[tool_name](**tool_args)
        return result
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return {"error": f"Tool '{tool_name}' raised an exception: {exc}"}


def run_agent_turn(client: genai.Client, user_message: str) -> str:
    """Runs one full agentic reasoning turn:
      1. Sends the user's message (+ history) to Gemini.
      2. If Gemini requests one or more tool calls, executes them locally.
      3. Sends the tool results back to Gemini.
      4. Repeats until Gemini returns a final natural-language answer or
         MAX_AGENT_TURNS is reached (safety cap against infinite loops).

    Returns the final text response to show the user. Tool call activity
    (including any forecast charts) is appended live to
    `st.session_state.display_messages` so the UI shows the agent's
    reasoning steps.
    """
    config = build_tool_config()

    st.session_state.chat_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    for _ in range(MAX_AGENT_TURNS):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=st.session_state.chat_history,
            config=config,
        )

        candidate = response.candidates[0]
        model_content = candidate.content
        st.session_state.chat_history.append(model_content)

        function_calls = [
            part.function_call
            for part in model_content.parts
            if getattr(part, "function_call", None) is not None
        ]

        if not function_calls:
            # No tool call requested -> this is the final answer.
            return response.text or "(The model returned an empty response.)"

        # Execute every requested tool call and build the corresponding
        # function response parts to send back to Gemini.
        response_parts = []
        for fc in function_calls:
            with st.status(f"🔧 Calling tool: `{fc.name}`", expanded=False) as status:
                st.write(f"Arguments: `{dict(fc.args or {})}`")
                tool_result = execute_tool_call(fc)
                st.write("Result:")
                st.json(tool_result)
                render_forecast_chart(tool_result)
                status.update(label=f"✅ Tool `{fc.name}` completed", state="complete")

            st.session_state.display_messages.append(
                {
                    "role": "tool",
                    "tool_name": fc.name,
                    "tool_args": dict(fc.args or {}),
                    "tool_result": tool_result,
                }
            )

            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": tool_result},
                )
            )

        st.session_state.chat_history.append(
            types.Content(role="user", parts=response_parts)
        )

    return (
        "I reached my reasoning step limit while using tools. Please "
        "rephrase your request or try again."
    )


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------
def render_history() -> None:
    """Renders the full chat history (user turns, assistant turns, and any
    intermediate tool-call traces with forecast charts) using
    st.chat_message bubbles."""
    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        elif msg["role"] == "tool":
            with st.chat_message("assistant", avatar="🔧"):
                st.markdown(f"**Tool used:** `{msg['tool_name']}`")
                st.caption(f"Arguments: {msg['tool_args']}")
                st.json(msg["tool_result"])
                render_forecast_chart(msg["tool_result"])


def render_sidebar() -> None:
    """Renders configuration/status info, tool list, and a reset button in
    the sidebar."""
    with st.sidebar:
        st.header("🚦 UrbanPulse")
        st.caption("AI Decision Intelligence for Smart Mobility")
        st.divider()

        st.markdown("**Session info**")
        st.markdown(f"- Model: `{MODEL_NAME}`")
        st.markdown(f"- Project: `{PROJECT_ID or 'NOT SET'}`")
        st.markdown(f"- Location: `{LOCATION}`")

        if not PROJECT_ID:
            st.warning(
                "GOOGLE_CLOUD_PROJECT is not set. Set it as an environment "
                "variable so the app can authenticate to Vertex AI."
            )

        st.divider()
        st.markdown("**Predictive tools available**")
        for fn in AVAILABLE_TOOLS:
            st.markdown(f"- `{fn.__name__}`")

        st.divider()
        st.markdown("**Try asking:**")
        st.markdown(
            "- \"Will Marine Drive be congested tomorrow at 6 PM?\"\n"
            "- \"Forecast ridership on Metro Line 3 for 2026-07-10.\"\n"
            "- \"Where can I find parking near Central Station this evening?\"\n"
            "- \"Is anything unusual expected in Downtown District tomorrow?\""
        )

        st.divider()
        if st.button("🗑️ Reset conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.display_messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("🚦 UrbanPulse — Smart Mobility Decision Intelligence")
    st.caption(
        "Powered by Gemini 2.5 Flash on Vertex AI, with autonomous predictive "
        "analytics for traffic, transit demand, parking, and mobility "
        "anomalies — built for the AI for Better Living and Smarter "
        "Communities challenge."
    )

    render_history()

    user_input = st.chat_input(
        "Ask about traffic, transit demand, parking, or anomalies..."
    )

    if user_input:
        st.session_state.display_messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing and forecasting..."):
                try:
                    client = get_genai_client()
                    final_answer = run_agent_turn(client, user_input)
                    st.markdown(final_answer)
                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": final_answer}
                    )
                except ClientError as exc:
                    error_msg = (
                        "⚠️ **Authentication or request error.** Please verify "
                        "your Google Cloud credentials, project ID, and that "
                        "the Vertex AI API is enabled.\n\n"
                        f"Details: `{exc}`"
                    )
                    st.error(error_msg)
                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                except APIError as exc:
                    error_msg = (
                        "⚠️ **The Gemini API returned an error** (this may be "
                        "a transient issue or a quota limit). Please try "
                        f"again in a moment.\n\nDetails: `{exc}`"
                    )
                    st.error(error_msg)
                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                except Exception as exc:  # noqa: BLE001
                    error_msg = (
                        "⚠️ **An unexpected error occurred.** Please try "
                        f"again.\n\nDetails: `{exc}`"
                    )
                    st.error(error_msg)
                    with st.expander("Show technical details"):
                        st.code(traceback.format_exc())
                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
