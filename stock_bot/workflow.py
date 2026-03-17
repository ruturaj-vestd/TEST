from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph

load_dotenv()


class AgentState(TypedDict, total=False):
    run_at: str
    shortlist: List[Dict[str, Any]]
    scout_notes: Dict[str, Any]
    analyst_notes: List[Dict[str, Any]]
    quant_notes: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    report_markdown: str


@dataclass
class BotConfig:
    min_market_cap: int = 2_000_000_000
    min_avg_volume: int = 1_000_000
    shortlist_size: int = 25
    lookback_period: str = "6mo"


def _get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your environment or .env file.")
    return genai.Client(api_key=api_key)


def _load_sp500_tickers() -> List[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    return [symbol.replace(".", "-") for symbol in df["Symbol"].tolist()]


def universe_filter_agent(state: AgentState, config: BotConfig) -> AgentState:
    tickers = _load_sp500_tickers()[:200]
    shortlisted = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get("marketCap") or 0
            avg_volume = info.get("averageVolume") or 0
            if market_cap >= config.min_market_cap and avg_volume >= config.min_avg_volume:
                shortlisted.append(
                    {
                        "ticker": ticker,
                        "market_cap": market_cap,
                        "average_volume": avg_volume,
                        "sector": info.get("sector", "Unknown"),
                    }
                )
        except Exception:
            continue

    shortlisted = sorted(shortlisted, key=lambda x: x["average_volume"], reverse=True)[: config.shortlist_size]
    state["shortlist"] = shortlisted
    return state


def scout_agent(state: AgentState, _: BotConfig) -> AgentState:
    client = _get_gemini_client()
    tickers = [item["ticker"] for item in state.get("shortlist", [])]

    prompt = (
        "You are a market scout. Using Google Search grounding, identify the strongest overnight or fresh catalysts "
        "for these tickers and return STRICT JSON with keys: ticker, catalyst, sentiment_score (-1 to 1), "
        "impact_level (1-10), source_hint. Tickers: "
        + ", ".join(tickers)
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
        ),
    )

    text = response.text or "[]"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
    except Exception:
        parsed = []

    by_ticker = {row["ticker"]: row for row in parsed if isinstance(row, dict) and row.get("ticker")}
    state["scout_notes"] = by_ticker
    return state


def analyst_agent(state: AgentState, _: BotConfig) -> AgentState:
    notes: List[Dict[str, Any]] = []

    for item in state.get("shortlist", []):
        ticker = item["ticker"]
        try:
            info = yf.Ticker(ticker).info
            quick_ratio = info.get("quickRatio")
            debt_to_equity = info.get("debtToEquity")
            pe_ratio = info.get("trailingPE")
            risk_flags = []
            if quick_ratio is not None and quick_ratio < 0.5:
                risk_flags.append("Low liquidity (quick ratio < 0.5)")
            if debt_to_equity is not None and debt_to_equity > 200:
                risk_flags.append("High leverage (debt-to-equity > 200)")

            notes.append(
                {
                    "ticker": ticker,
                    "quick_ratio": quick_ratio,
                    "debt_to_equity": debt_to_equity,
                    "pe_ratio": pe_ratio,
                    "risk_flags": risk_flags,
                }
            )
        except Exception:
            continue

    state["analyst_notes"] = notes
    return state


def quant_agent(state: AgentState, config: BotConfig) -> AgentState:
    quant_rows: List[Dict[str, Any]] = []
    for item in state.get("shortlist", []):
        ticker = item["ticker"]
        hist = yf.Ticker(ticker).history(period=config.lookback_period, interval="1d")
        if hist.empty or len(hist) < 30:
            continue

        close = hist["Close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()

        quant_rows.append(
            {
                "ticker": ticker,
                "last_close": float(close.iloc[-1]),
                "rsi14": None if pd.isna(rsi.iloc[-1]) else float(rsi.iloc[-1]),
                "ma20": None if pd.isna(ma20.iloc[-1]) else float(ma20.iloc[-1]),
                "ma50": None if pd.isna(ma50.iloc[-1]) else float(ma50.iloc[-1]),
            }
        )

    state["quant_notes"] = quant_rows
    return state


def predictor_agent(state: AgentState, _: BotConfig) -> AgentState:
    client = _get_gemini_client()

    fundamentals = {row["ticker"]: row for row in state.get("analyst_notes", [])}
    quant = {row["ticker"]: row for row in state.get("quant_notes", [])}
    scout = state.get("scout_notes", {})

    enriched = []
    for item in state.get("shortlist", []):
        ticker = item["ticker"]
        if ticker in quant:
            enriched.append(
                {
                    "ticker": ticker,
                    "market": item,
                    "scout": scout.get(ticker, {}),
                    "fundamentals": fundamentals.get(ticker, {}),
                    "quant": quant.get(ticker, {}),
                }
            )

    prompt = (
        "You are Predictor Agent. Build probabilistic open/close forecasts for next trading day. "
        "Use only provided numbers for calculations and mention why. Return STRICT JSON list with: "
        "ticker, direction (bullish/bearish/neutral), predicted_open_change_pct, predicted_close_change_pct, "
        "confidence_0_to_1, reasoning, stop_loss_pct. Input: "
        + json.dumps(enriched)
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
    )
    text = response.text or "[]"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
    except Exception:
        parsed = []

    state["predictions"] = [row for row in parsed if isinstance(row, dict)]
    return state


def report_agent(state: AgentState, _: BotConfig) -> AgentState:
    now = state.get("run_at")
    lines = [
        f"# Daily Multi-Agent Stock Report ({now})",
        "",
        "## Top Forecasts",
    ]

    rows = sorted(state.get("predictions", []), key=lambda x: x.get("confidence_0_to_1", 0), reverse=True)[:6]
    for row in rows:
        lines.append(
            f"- **{row.get('ticker','N/A')}** | {row.get('direction','n/a')} | "
            f"Open: {row.get('predicted_open_change_pct','?')}% | "
            f"Close: {row.get('predicted_close_change_pct','?')}% | "
            f"Confidence: {row.get('confidence_0_to_1','?')}"
        )
        lines.append(f"  - Why: {row.get('reasoning','No rationale provided')}.")
        lines.append(f"  - Suggested stop loss: {row.get('stop_loss_pct','N/A')}%")

    lines.extend(
        [
            "",
            "## Risk Notes",
            "- LLM output is probabilistic and can be wrong; do not treat this as financial advice.",
            "- Stop losses are suggestions for monitoring automation, not guaranteed protections.",
        ]
    )

    state["report_markdown"] = "\n".join(lines)
    return state


def build_graph() -> Any:
    config = BotConfig()
    graph = StateGraph(AgentState)
    graph.add_node("universe_filter", lambda s: universe_filter_agent(s, config))
    graph.add_node("scout", lambda s: scout_agent(s, config))
    graph.add_node("analyst", lambda s: analyst_agent(s, config))
    graph.add_node("quant", lambda s: quant_agent(s, config))
    graph.add_node("predictor", lambda s: predictor_agent(s, config))
    graph.add_node("report", lambda s: report_agent(s, config))

    graph.set_entry_point("universe_filter")
    graph.add_edge("universe_filter", "scout")
    graph.add_edge("scout", "analyst")
    graph.add_edge("analyst", "quant")
    graph.add_edge("quant", "predictor")
    graph.add_edge("predictor", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_daily(output_dir: str = "reports") -> Path:
    app = build_graph()
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result = app.invoke({"run_at": run_at})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"report_{datetime.now().strftime('%Y%m%d')}.md"
    out_file.write_text(result.get("report_markdown", "No report generated."), encoding="utf-8")
    return out_file
