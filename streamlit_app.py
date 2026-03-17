import json
import os

import pandas as pd
import streamlit as st

from indian_stock_pipeline import run_pipeline

st.set_page_config(page_title="Indian Stock Multi-Agent Bot", page_icon="📈", layout="wide")

st.title("📈 Indian Stock Multi-Agent Forecast Bot")
st.caption("LangGraph + Gemini + free market/news data sources")

with st.sidebar:
    st.header("Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Required. Get from Google AI Studio.",
    )
    telegram_bot_token = st.text_input(
        "Telegram Bot Token (optional)",
        type="password",
        value=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    )
    telegram_chat_id = st.text_input(
        "Telegram Chat ID (optional)",
        value=os.getenv("TELEGRAM_CHAT_ID", ""),
    )
    run_clicked = st.button("Run Daily Analysis", type="primary", use_container_width=True)

st.markdown(
    """
### Workflow
1. **Scout Agent** scans RSS + sentiment impact.
2. **Analyst Agent** validates fundamentals (P/E, debt, quick ratio).
3. **Quant Agent** computes RSI/SMA/volatility from OHLC.
4. **Predictor Agent** generates open/close probabilistic forecast + rationale.
5. **Risk output** includes stop-loss and take-profit suggestions.
"""
)

if run_clicked:
    if not gemini_api_key:
        st.error("Gemini API key is required.")
    else:
        with st.spinner("Running full multi-agent pipeline. This can take a few minutes..."):
            try:
                result = run_pipeline(
                    gemini_api_key=gemini_api_key,
                    telegram_bot_token=telegram_bot_token or None,
                    telegram_chat_id=telegram_chat_id or None,
                )
            except Exception as exc:
                st.exception(exc)
                st.stop()

        report = result.get("final_report", {})
        output_paths = result.get("output_paths", {})

        st.success("Analysis completed.")

        top_picks = report.get("top_picks", [])
        all_predictions = report.get("all_predictions", [])

        if top_picks:
            st.subheader("Top Picks")
            top_df = pd.DataFrame(top_picks)
            st.dataframe(top_df, use_container_width=True)

            for i, row in enumerate(top_picks, start=1):
                with st.expander(f"{i}. {row.get('symbol')} | {row.get('direction')} | conf={row.get('confidence')}"):
                    st.write(row.get("why", "No explanation available."))
                    st.json(row)
        else:
            st.warning("No top picks generated.")

        st.subheader("All Predictions")
        st.dataframe(pd.DataFrame(all_predictions), use_container_width=True)

        st.subheader("Saved Files")
        st.code(json.dumps(output_paths, indent=2))

        json_path = output_paths.get("json")
        if json_path and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as fh:
                payload = fh.read()
            st.download_button(
                "Download JSON Report",
                data=payload,
                file_name=os.path.basename(json_path),
                mime="application/json",
            )

        md_path = output_paths.get("md")
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as fh:
                md_payload = fh.read()
            st.download_button(
                "Download Markdown Report",
                data=md_payload,
                file_name=os.path.basename(md_path),
                mime="text/markdown",
            )
else:
    st.info("Add your Gemini key in the sidebar and click **Run Daily Analysis**.")
