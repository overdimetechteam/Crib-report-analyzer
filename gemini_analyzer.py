#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_analyzer.py
==================
Gemini-powered credit risk analysis using LangChain and LangGraph.
Provides table-based tool calling to enhance risk assessment insights.

USAGE
-----
    from gemini_analyzer import analyze_with_gemini
    enhanced_risk = analyze_with_gemini(parsed, analysis, risk)

REQUIREMENTS
------------
    pip install langchain langchain-google-genai langgraph

CONFIGURATION
-------------
    Reads google_api_key / gemini_model / gemini_temperature / gemini_max_tokens
    from a .env file (KEY=value per line) in the current working directory,
    or from the matching GOOGLE_API_KEY / GEMINI_MODEL / ... environment
    variables, which take precedence over .env.
"""
import json
import os
import re
from pathlib import Path
from typing import Annotated, TypedDict, TYPE_CHECKING

LANGCHAIN_AVAILABLE = False
if TYPE_CHECKING:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

try:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


class AnalysisState(TypedDict):
    """State container for LangGraph workflow."""
    messages: Annotated[list, add_messages]
    analysis_done: bool


def _load_env_file(path='.env'):
    """Parse a simple KEY=value .env file into a dict of lowercase keys."""
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            env[key.strip().lower()] = value.strip().strip('"\'')
    return env


def get_gemini_settings():
    """Resolve Gemini settings from .env, with matching env vars taking precedence."""
    env = _load_env_file()
    return {
        'google_api_key': os.getenv('GOOGLE_API_KEY') or env.get('google_api_key', ''),
        'gemini_model': os.getenv('GEMINI_MODEL') or env.get('gemini_model') or 'gemini-2.0-flash',
        'gemini_temperature': float(os.getenv('GEMINI_TEMPERATURE') or env.get('gemini_temperature') or 0.3),
        'gemini_max_tokens': int(os.getenv('GEMINI_MAX_TOKENS') or env.get('gemini_max_tokens') or 1024),
    }


def is_gemini_enabled():
    """True if a google_api_key is configured via .env or environment variables."""
    return bool(get_gemini_settings()['google_api_key'].strip())


def build_facility_table_rows(parsed):
    """Extract and format facility table rows from parsed CRIB data."""
    rows = []
    for idx in sorted(parsed['facilities']):
        f = parsed['facilities'][idx]
        ownership = (f.get('ownership') or '').strip()
        if ownership not in ('OWN', 'JNT', 'GRT'):
            continue
        status = (f.get('status') or '').strip()
        wo = f.get('written_off') or 0
        if wo > 0:
            pill, pill_cls = 'WRITTEN OFF', 'pill-written'
        elif status == 'ACTV':
            pill, pill_cls = 'ACTIVE', 'pill-active'
        elif status == 'SETL':
            pill, pill_cls = 'SETTLED', 'pill-settled'
        else:
            continue
        rows.append({
            'institution': f.get('member') or f.get('institution_category') or '-',
            'type': (f.get('type') or '-') + (' (Guarantor)' if ownership == 'GRT' else ''),
            'granted': f.get('granted'),
            'os': f.get('balance'),
            'pill': pill,
            'pill_class': pill_cls,
            'flag': wo > 0,
        })
    return rows


def analyze_with_gemini(parsed, analysis, risk, model=None):
    """
    Use Gemini with table tools to enhance red flags, positives, and recommendations.
    
    Args:
        parsed: Parsed CRIB report data
        analysis: Analysis results (portfolio, arrears, etc.)
        risk: Risk assessment dict
        model: Gemini model name (default: from config)
        
    Returns:
        Enhanced risk dict with Gemini-generated insights appended to red_flags, positives, and recommendation
    """
    settings = get_gemini_settings()

    if not settings['google_api_key'] or not LANGCHAIN_AVAILABLE:
        return risk

    api_key = settings['google_api_key']
    if not model:
        model = settings['gemini_model']
    temperature = settings['gemini_temperature']

    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

        @tool
        def get_facility_table() -> str:
            """Return the facility table rows (institution, type, granted, outstanding, status)."""
            rows = build_facility_table_rows(parsed)
            return json.dumps(rows[:10], ensure_ascii=False, indent=2)

        @tool
        def get_portfolio_breakdown() -> str:
            """Return portfolio breakdown metrics by facility type and repayment method."""
            return json.dumps(analysis.get('portfolio', {}), ensure_ascii=False, indent=2)

        @tool
        def get_arrears_detail() -> str:
            """Return arrears ageing buckets and total delinquency amounts."""
            arr = analysis.get('arrears', {})
            return json.dumps({'total': arr.get('total'), 'buckets': arr.get('buckets', [])}, ensure_ascii=False, indent=2)

        @tool
        def get_score_trend() -> str:
            """Return recent CRIB score history (last 6 months)."""
            scores = risk.get('scores', [])
            return json.dumps(scores[-6:], ensure_ascii=False, indent=2)

        tools = [get_facility_table, get_portfolio_breakdown, get_arrears_detail, get_score_trend]
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state):
            response = llm_with_tools.invoke(state['messages'])
            return {'messages': [response], 'analysis_done': False}

        def call_tool(state):
            last_message = state['messages'][-1]
            tool_calls = getattr(last_message, 'tool_calls', []) or []
            if not tool_calls:
                return {'messages': [], 'analysis_done': True}

            tool_messages = []
            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})
                tool_id = tool_call.get('id')
                if not tool_name or not tool_id:
                    continue
                tool_fn = next((t for t in tools if t.name == tool_name), None)
                if tool_fn is None:
                    tool_output = f'Unknown tool: {tool_name}'
                else:
                    try:
                        tool_output = tool_fn.invoke(tool_args)
                    except Exception as exc:
                        tool_output = f'Tool error: {exc}'
                tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id, name=tool_name))
            return {'messages': tool_messages, 'analysis_done': False}

        def should_continue(state):
            last_message = state['messages'][-1]
            if getattr(last_message, 'tool_calls', None):
                return 'tools'
            return 'end'

        workflow = StateGraph(AnalysisState)
        workflow.add_node('model', call_model)
        workflow.add_node('tools', call_tool)
        workflow.add_edge(START, 'model')
        workflow.add_conditional_edges('model', should_continue, {'tools': 'tools', 'end': END})
        workflow.add_edge('tools', 'model')
        app = workflow.compile()

        initial_messages = [
            SystemMessage(content=(
                'You are a credit-risk analyst. Analyze the report tables to provide enhanced insights.\n'
                'Respond with a JSON object containing ONLY:\n'
                '  {"red_flags": ["flag1", "flag2", ...], "positives": ["positive1", ...], "recommendation": "short recommendation text"}\n'
                'Be specific and concise. Do NOT include any text outside the JSON object.'
            )),
            HumanMessage(content=(
                'Analyze this credit report and identify red flags, positive factors, and provide a concise recommendation. '
                'Use the available tools to inspect the facility, portfolio, arrears, and score trends.\n\n'
                f'Current assessment: decision={risk.get("decision")}, risk_level={risk.get("risk_level")}'
            )),
        ]
        result = app.invoke({'messages': initial_messages, 'analysis_done': False})
        
        final_text = None
        for message in reversed(result.get('messages', [])):
            content = getattr(message, 'content', None)
            if not content or getattr(message, 'tool_calls', None):
                continue
            if isinstance(content, list):
                content = ''.join(
                    part.get('text', '') for part in content
                    if isinstance(part, dict) and part.get('type') == 'text'
                )
            final_text = (content or '').strip()
            if final_text:
                break
        
        if not final_text:
            return risk
        
        try:
            json_match = re.search(r'\{.*\}', final_text, re.DOTALL)
            if json_match:
                gemini_result = json.loads(json_match.group())
            else:
                gemini_result = json.loads(final_text)
            
            if gemini_result.get('red_flags'):
                risk['red_flags'] = risk.get('red_flags', []) + gemini_result['red_flags']
            if gemini_result.get('positives'):
                risk['positives'] = risk.get('positives', []) + gemini_result['positives']
            if gemini_result.get('recommendation'):
                risk['recommendation'] = gemini_result['recommendation']
            
            return risk
        except (json.JSONDecodeError, AttributeError):
            return risk
    except Exception:
        return risk
