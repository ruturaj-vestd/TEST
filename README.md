# Multi-Agent Daily Stock Forecast Bot (LangGraph + Gemini)

This project implements a **multi-agent stock analysis factory** using only free-access tooling and a single required credential: `GEMINI_API_KEY`.

## What it does

Every run builds a shortlist, analyzes catalysts + fundamentals + technicals, then produces a probabilistic open/close forecast with rationale.

### Agents

1. **Universe Filter Agent**
   - Pulls S&P 500 symbols.
   - Filters by:
     - `market_cap > $2B`
     - `average_volume > 1M`
   - Keeps top ~25 liquid names.

2. **Scout Agent**
   - Uses **Gemini + Google Search grounding** to detect fresh catalysts/news.
   - Assigns sentiment and impact level.

3. **Analyst Agent**
   - Pulls fundamentals from free Yahoo Finance data:
     - P/E (`trailingPE`)
     - Debt-to-equity
     - Quick ratio
   - Flags risky balance-sheet profiles.

4. **Quant Agent**
   - Pulls historical OHLC.
   - Computes:
     - RSI(14)
     - MA20
     - MA50

5. **Predictor Agent (Brain)**
   - Gemini synthesizes scout + analyst + quant outputs.
   - Produces probabilistic forecasts:
     - next open change %
     - next close change %
     - confidence
     - “why” explanation
     - suggested stop-loss %

6. **Report Agent**
   - Writes markdown report under `reports/`.

---

## Step-by-step implementation (mapped to your requested flow)

### Step A: Filter the universe
Implemented in `universe_filter_agent()`:
- gets S&P 500 list
- filters by market cap + volume
- narrows to manageable shortlist

### Step B: Sentiment and news context
Implemented in `scout_agent()`:
- Gemini grounded web search classifies catalysts
- includes impact level and sentiment score

### Step C: Fundamental validation
Implemented in `analyst_agent()`:
- cross-checks quick ratio and debt-to-equity
- adds risk flags when hype != financial health

### Step D: Probabilistic open/close prediction
Implemented in `predictor_agent()`:
- generates probabilistic open/close forecasts
- provides explicit “why” narrative
- includes stop-loss suggestion for risk monitoring

---

## Why this is free-friendly

- ✅ **Only key required:** `GEMINI_API_KEY`
- ✅ Market/fundamental data from Yahoo Finance (`yfinance`), no paid key required
- ✅ Scheduled daily runs via local scheduler mode or GitHub Actions cron

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_key_here
```

---

## Run once

```bash
python -m stock_bot.cli --output-dir reports
```

---

## Run daily (local scheduler)

```bash
python -m stock_bot.cli --schedule --run-time-utc 12:30 --output-dir reports
```

---

## Daily automation via GitHub Actions

Workflow file: `.github/workflows/daily_report.yml`

- Runs every day at 12:30 UTC
- Also supports manual trigger
- Uploads markdown report artifact

Set repository secret:
- `GEMINI_API_KEY`

---

## Important limitations and safeguards

- LLM is **not used for raw financial calculations**; Python does calculations first.
- Forecasts are probabilistic; markets are stochastic.
- This is **not financial advice**.
- Prefer official APIs and compliant data usage in production.
