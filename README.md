# Indian Stock Multi-Agent Forecast Bot (LangGraph + Streamlit)

This repo contains a multi-agent stock-analysis workflow for the Indian market using only free data sources and a Gemini API key.

## Agents

- **Scout Agent**: Reads business/market RSS feeds and uses Gemini sentiment/impact scoring.
- **Analyst Agent**: Pulls fundamentals from Yahoo Finance (`yfinance`) and flags weak health.
- **Quant Agent**: Pulls OHLC data and computes RSI, moving averages, and ATR-like volatility.
- **Predictor Agent**: Uses Gemini to produce probabilistic open/close move forecasts and explain the reasons.
- **Report Agent**: Produces JSON + Markdown reports and optional Telegram summary.

## Project Files

- `indian_stock_pipeline.py` – LangGraph pipeline and all agent nodes.
- `streamlit_app.py` – Streamlit UI.
- `requirements.txt` – dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Streamlit Interface

```bash
streamlit run streamlit_app.py
```

In the UI:
1. Enter `Gemini API Key` in sidebar.
2. (Optional) Add Telegram bot token + chat ID.
3. Click **Run Daily Analysis**.
4. View Top Picks, Full Table, and download JSON/Markdown reports.

## CLI Run (without UI)

```bash
export GEMINI_API_KEY="your_key"
python indian_stock_pipeline.py
```

Reports are generated under `reports/`.

## Daily Auto-Update (Cron)

Use cron to run every weekday morning:

```cron
0 7 * * 1-5 cd /path/to/repo && /path/to/repo/.venv/bin/python indian_stock_pipeline.py >> bot.log 2>&1
```

## Notes

- This is **not financial advice**.
- LLMs are used for interpretation, not raw financial calculations.
- Uses public feeds/APIs and avoids paid-only integrations except Gemini API key.
