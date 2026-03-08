"""
Agent Service — Orchestrates the three AI analyst agents and the consensus step.

Each agent:
  1. Receives the top-K most relevant document chunks (from RAG retrieval)
  2. Is prompted with its distinct investment philosophy
  3. Returns a structured JSON verdict grounded in the document

The consensus step synthesises all three verdicts into a committee decision.
"""

import asyncio
import json
from typing import Optional
from openai import AsyncOpenAI

from app.core.config import settings

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

AGENTS = [
    {
        "id": "graham",
        "name": "Benjamin Graham",
        "title": "Value Analyst",
        "philosophy": (
            "You are Benjamin Graham, father of value investing. "
            "You focus obsessively on intrinsic value vs market price, margin of safety, "
            "balance sheet strength, earnings stability, and avoiding speculation. "
            "You are skeptical of growth narratives and demand a discount to fair value."
        ),
    },
    {
        "id": "wood",
        "name": "Cathie Wood",
        "title": "Growth Analyst",
        "philosophy": (
            "You are Cathie Wood, a high-conviction disruptive innovation investor. "
            "You focus on exponential growth curves, total addressable market expansion, "
            "platform businesses, AI/tech adoption, and 5-year price targets. "
            "You look for companies compounding at 15%+ annually."
        ),
    },
    {
        "id": "risk",
        "name": "Risk Committee",
        "title": "Risk Manager",
        "philosophy": (
            "You are the Risk Committee chair. "
            "You focus exclusively on downside risk: concentration risk, regulatory exposure, "
            "macro headwinds, liquidity, leverage, competitive threats, and tail scenarios. "
            "Your job is to challenge the bull case and protect capital."
        ),
    },
]

# ---------------------------------------------------------------------------
# Shared OpenAI client
# ---------------------------------------------------------------------------

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Single agent analysis
# ---------------------------------------------------------------------------

async def run_agent(
    agent: dict,
    ticker: str,
    context_chunks: list[str],
) -> dict:
    """
    Run one analyst agent against the retrieved document chunks.
    Returns a structured dict with verdict, conviction, reasoning etc.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""{agent['philosophy']}

You are analysing {ticker}. Below are the most relevant excerpts from the financial document, retrieved specifically for your analytical lens:

===DOCUMENT EXCERPTS===
{context}
===END EXCERPTS===

Based ONLY on the document excerpts above, provide your investment assessment.
Respond with a single valid JSON object and nothing else — no markdown fences, no preamble:

{{
  "verdict": "BUY or HOLD or SELL",
  "conviction": <integer 1-10>,
  "price_target_commentary": "<one sentence on valuation from the document>",
  "bull_case": "<2-3 sentences on the strongest positive signal in the document>",
  "bear_case": "<2-3 sentences on the biggest risk evident in the document>",
  "key_quote": "<the single most important data point from the document driving your view>",
  "reasoning": "<3-4 sentences of your overall thesis in your voice and philosophy>"
}}"""

    try:
        message = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=settings.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "verdict": None, "conviction": None}


# ---------------------------------------------------------------------------
# Consensus generation
# ---------------------------------------------------------------------------

async def generate_consensus(ticker: str, agent_results: dict[str, dict]) -> dict:
    """
    Synthesise individual agent verdicts into a committee consensus.
    """
    summaries = []
    for agent in AGENTS:
        r = agent_results.get(agent["id"], {})
        if r.get("verdict"):
            summaries.append(
                f"{agent['name']} ({agent['title']}): {r['verdict']} | "
                f"Conviction {r.get('conviction')}/10 | {r.get('reasoning', '')}"
            )
        else:
            summaries.append(f"{agent['name']}: Analysis unavailable")

    prompt = f"""Three investment analysts have independently reviewed {ticker}.
Synthesise their views into a committee consensus.
Respond with a single valid JSON object only — no markdown, no preamble:

{chr(10).join(summaries)}

{{
  "committee_verdict": "BUY or HOLD or SELL",
  "confidence": "HIGH or MEDIUM or LOW",
  "agreement_level": "UNANIMOUS or MAJORITY or SPLIT",
  "summary": "<3-4 sentence synthesis of the committee view>",
  "key_risk": "<single biggest risk to flag>",
  "key_opportunity": "<single biggest opportunity identified>"
}}"""

    try:
        message = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Orchestrator — runs all agents in parallel then generates consensus
# ---------------------------------------------------------------------------

async def run_committee(
    ticker: str,
    context_chunks: list[str],
) -> dict:
    """
    Run all three agents concurrently, then generate the committee consensus.
    Returns a dict: { "agents": {...}, "consensus": {...} }
    """
    # Run all agents in parallel with asyncio.gather
    tasks = [run_agent(agent, ticker, context_chunks) for agent in AGENTS]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    agent_results = {}
    for agent, result in zip(AGENTS, results_list):
        if isinstance(result, Exception):
            agent_results[agent["id"]] = {"error": str(result)}
        else:
            agent_results[agent["id"]] = result

    # Generate consensus from all agent results
    consensus = await generate_consensus(ticker, agent_results)

    return {
        "agents": agent_results,
        "consensus": consensus,
    }
