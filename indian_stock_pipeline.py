import datetime as dt
import json
import math
import os
from typing import Any, Dict, List, TypedDict

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph

NIFTY_50_NS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HINDUNILVR.NS",
    "SUNPHARMA.NS", "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "BAJFINANCE.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "TECHM.NS", "M&M.NS",
]

RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/business.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
]


class PipelineState(TypedDict, total=False):
    run_date: str
    universe: List[str]
    screened: List[str]
    news_by_symbol: Dict[str, List[Dict[str, str]]]
    news_scores: Dict[str, Dict[str, Any]]
    fundamentals: Dict[str, Dict[str, Any]]
    technicals: Dict[str, Dict[str, Any]]
    predictions: Dict[str, Dict[str, Any]]
    final_report: Dict[str, Any]
    output_paths: Dict[str, str]


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return np.nan
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan


def _compute_atr_like(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return np.nan
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1]) if not atr.empty else np.nan


def _clean_news_text(text: str) -> str:
    return " ".join((text or "").split())[:1000]


def _gemini_generate_json(api_key: str, prompt: str, use_google_search: bool = True) -> Dict[str, Any]:
    client = genai.Client(api_key=api_key)
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_google_search else None

    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            tools=tools,
        ),
    )

    txt = response.text.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"raw_text": txt, "parse_error": True}


def _send_telegram_message(message: str, token: str | None, chat_id: str | None) -> None:
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message[:4000]}, timeout=20)


def universe_node(state: PipelineState) -> PipelineState:
    return {
        "universe": NIFTY_50_NS,
        "run_date": dt.datetime.now().strftime("%Y-%m-%d"),
    }


def scout_node(state: PipelineState) -> PipelineState:
    universe = state["universe"]
    api_key = state["gemini_api_key"]  # type: ignore[index]

    all_entries = []
    for url in RSS_FEEDS:
        parsed = feedparser.parse(url)
        for e in parsed.entries[:80]:
            all_entries.append(
                {
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "link": getattr(e, "link", ""),
                    "published": getattr(e, "published", ""),
                }
            )

    symbol_to_news = {s: [] for s in universe}
    simple_names = {s: s.split(".")[0] for s in universe}

    for item in all_entries:
        blob = (item["title"] + " " + item["summary"]).upper()
        for symbol, root in simple_names.items():
            if root in blob:
                symbol_to_news[symbol].append(item)

    news_scores: Dict[str, Dict[str, Any]] = {}
    screened: List[str] = []

    for symbol in universe:
        items = symbol_to_news[symbol][:5]
        headline_blob = "\n".join(
            [f"- {_clean_news_text(i['title'])} | {_clean_news_text(i['summary'])}" for i in items]
        ) or "No direct stock-specific headlines found."

        prompt = f"""
You are a financial news scout for Indian equities.
Stock: {symbol}
News:
{headline_blob}

Return strict JSON with:
- sentiment_score: float in [-1,1]
- impact_score: int in [1,10]
- catalyst_type: one of ["earnings","guidance","management","regulatory","macro","rumor","none"]
- rationale: short string
"""
        scored = _gemini_generate_json(api_key, prompt, use_google_search=True)
        news_scores[symbol] = scored

        impact = scored.get("impact_score", 1)
        sentiment = scored.get("sentiment_score", 0.0)
        if isinstance(impact, (int, float)) and (impact >= 5 or abs(float(sentiment)) >= 0.3):
            screened.append(symbol)

    screened = screened[:30]
    if len(screened) < 20:
        extra = [s for s in universe if s not in screened][: 20 - len(screened)]
        screened.extend(extra)

    return {
        "news_by_symbol": symbol_to_news,
        "news_scores": news_scores,
        "screened": screened,
    }


def analyst_node(state: PipelineState) -> PipelineState:
    fundamentals: Dict[str, Dict[str, Any]] = {}

    for symbol in state["screened"]:
        t = yf.Ticker(symbol)
        info = t.info if isinstance(t.info, dict) else {}
        fast = t.fast_info if hasattr(t, "fast_info") else {}

        market_cap_fallback = fast.get("market_cap") if isinstance(fast, dict) else None
        fundamentals[symbol] = {
            "market_cap": _safe_float(info.get("marketCap", market_cap_fallback)),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "quick_ratio": _safe_float(info.get("quickRatio")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "roe": _safe_float(info.get("returnOnEquity")),
            "flag_health_risk": False,
        }

        qr = fundamentals[symbol]["quick_ratio"]
        dte = fundamentals[symbol]["debt_to_equity"]
        if (not math.isnan(qr) and qr < 0.5) or (not math.isnan(dte) and dte > 200):
            fundamentals[symbol]["flag_health_risk"] = True

    return {"fundamentals": fundamentals}


def quant_node(state: PipelineState) -> PipelineState:
    technicals: Dict[str, Dict[str, Any]] = {}

    for symbol in state["screened"]:
        df = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty or len(df) < 60:
            technicals[symbol] = {"data_error": True}
            continue

        close = df["Close"]
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan

        technicals[symbol] = {
            "last_close": float(close.iloc[-1]),
            "rsi14": _compute_rsi(close, 14),
            "sma20": sma20,
            "sma50": sma50,
            "ret_5d_pct": float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else np.nan,
            "atr14": _compute_atr_like(df, 14),
            "bullish_ma_cross": bool(sma20 > sma50) if not (math.isnan(sma20) or math.isnan(sma50)) else False,
        }

    return {"technicals": technicals}


def predictor_node(state: PipelineState) -> PipelineState:
    api_key = state["gemini_api_key"]  # type: ignore[index]
    predictions: Dict[str, Dict[str, Any]] = {}

    for symbol in state["screened"]:
        fundamentals = state["fundamentals"].get(symbol, {})
        technicals = state["technicals"].get(symbol, {})
        news_score = state["news_scores"].get(symbol, {})
        top_news = state["news_by_symbol"].get(symbol, [])[:3]

        prompt = f"""
You are the Predictor Agent for Indian stocks.
Generate a probabilistic next-session forecast for {symbol}.

INPUT:
Fundamentals: {json.dumps(fundamentals, default=str)}
Technicals: {json.dumps(technicals, default=str)}
NewsScore: {json.dumps(news_score, default=str)}
TopNews: {json.dumps(top_news, default=str)}

Rules:
- Do NOT invent raw metrics.
- Use provided numbers as facts.
- Output strict JSON keys:
  - open_move_pct_mean (float, expected % move at open)
  - open_move_pct_range (string)
  - close_move_pct_mean (float)
  - close_move_pct_range (string)
  - confidence (0-1 float)
  - direction ("bullish"|"bearish"|"neutral")
  - why (2-4 sentence explanation)
  - risk_flags (array of strings)
  - suggested_stop_loss_pct (float)
  - suggested_take_profit_pct (float)
"""
        predictions[symbol] = _gemini_generate_json(api_key, prompt, use_google_search=False)

    return {"predictions": predictions}


def report_node(state: PipelineState) -> PipelineState:
    rows: List[Dict[str, Any]] = []
    for symbol in state["screened"]:
        prediction = state["predictions"].get(symbol, {})
        score = state["news_scores"].get(symbol, {})
        fundamentals = state["fundamentals"].get(symbol, {})
        technicals = state["technicals"].get(symbol, {})

        rows.append(
            {
                "symbol": symbol,
                "direction": prediction.get("direction"),
                "confidence": prediction.get("confidence"),
                "open_mean_pct": prediction.get("open_move_pct_mean"),
                "close_mean_pct": prediction.get("close_move_pct_mean"),
                "impact_score": score.get("impact_score"),
                "sentiment_score": score.get("sentiment_score"),
                "health_risk": fundamentals.get("flag_health_risk"),
                "rsi14": technicals.get("rsi14"),
                "why": prediction.get("why"),
            }
        )

    def ranking(item: Dict[str, Any]) -> float:
        try:
            return abs(float(item.get("close_mean_pct", 0))) * float(item.get("confidence", 0))
        except Exception:
            return 0.0

    ranked = sorted(rows, key=ranking, reverse=True)
    top = ranked[:6]

    report = {
        "run_date": state["run_date"],
        "top_picks": top,
        "all_predictions": ranked,
    }

    os.makedirs("reports", exist_ok=True)
    date_part = state["run_date"]
    json_path = f"reports/report_{date_part}.json"
    md_path = f"reports/report_{date_part}.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    lines = [f"# Daily Indian Stock Agent Report ({date_part})", ""]
    for idx, item in enumerate(top, start=1):
        lines.extend(
            [
                f"## {idx}. {item['symbol']}",
                f"- Direction: **{item.get('direction')}**",
                f"- Confidence: **{item.get('confidence')}**",
                f"- Expected Open Move (%): **{item.get('open_mean_pct')}**",
                f"- Expected Close Move (%): **{item.get('close_mean_pct')}**",
                f"- News Impact: **{item.get('impact_score')}**, Sentiment: **{item.get('sentiment_score')}**",
                f"- Health Risk Flag: **{item.get('health_risk')}**",
                f"- RSI14: **{item.get('rsi14')}**",
                f"- Why: {item.get('why')}",
                "",
            ]
        )

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    _send_telegram_message(
        message="Daily Stock Bot picks:\n"
        + "\n".join([f"{x['symbol']} | {x.get('direction')} | conf {x.get('confidence')}" for x in top[:5]]),
        token=state.get("telegram_bot_token"),  # type: ignore[arg-type]
        chat_id=state.get("telegram_chat_id"),  # type: ignore[arg-type]
    )

    return {
        "final_report": report,
        "output_paths": {"json": json_path, "md": md_path},
    }


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("universe", universe_node)
    graph.add_node("scout", scout_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("quant", quant_node)
    graph.add_node("predictor", predictor_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("universe")
    graph.add_edge("universe", "scout")
    graph.add_edge("scout", "analyst")
    graph.add_edge("analyst", "quant")
    graph.add_edge("quant", "predictor")
    graph.add_edge("predictor", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(gemini_api_key: str, telegram_bot_token: str | None = None, telegram_chat_id: str | None = None) -> Dict[str, Any]:
    app = build_graph()
    initial_state: PipelineState = {
        "gemini_api_key": gemini_api_key,  # type: ignore[typeddict-item]
        "telegram_bot_token": telegram_bot_token,  # type: ignore[typeddict-item]
        "telegram_chat_id": telegram_chat_id,  # type: ignore[typeddict-item]
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY before running.")
    result = run_pipeline(
        gemini_api_key=key,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    )
    print("Done", result.get("output_paths"))
