import os
import datetime as dt
import streamlit as st
import pandas as pd

from src.timeseries import load_prices, features_summary, plot_prices
from src.news import fetch_news_rss
from src.ingest import ensure_sec_index
from src.retriever import retrieve
from src.rag import generate_insights, qa_answer
from src.risk import risk_report

st.set_page_config(page_title="Financial RAG w/ Time-Series", layout="wide")

st.title("📈 Financial Analysis RAG with Time‑Series Data")
st.caption("Educational demo — not investment advice.")

with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    start_days = st.number_input("Lookback (days)", min_value=30, max_value=1825, value=365, step=5)
    start_date = dt.date.today() - dt.timedelta(days=int(start_days))
    end_date = dt.date.today()
    build = st.button("🔎 Build/Refresh SEC Index")
    st.divider()
    st.write("LLM: OpenAI via `OPENAI_API_KEY`")

if build:
    with st.spinner("Building SEC filings index (10-K/10-Q) in Chroma…"):
        msg = ensure_sec_index(ticker)
        st.success(msg)

# === TIME SERIES ===
prices = load_prices(ticker, start_date, end_date)
if prices is None or prices.empty:
    st.warning("No price data found. Try another ticker.")
    st.stop()

col1, col2 = st.columns([2,1])
with col1:
    st.subheader(f"Price & Trend • {ticker}")
    fig = plot_prices(prices)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Feature Snapshot")
    st.dataframe(features_summary(prices))

st.divider()

# === NEWS ===
st.subheader(f"📰 Latest News • {ticker}")
news_items = fetch_news_rss(ticker, limit=10)
for n in news_items:
    st.markdown(f"- [{n['title']}]({n['link']}) — {n['published']}")

st.divider()

# === INSIGHTS ===
st.subheader("💡 Insights (RAG + Time-Series + News)")
with st.spinner("Generating insights…"):
    insights = generate_insights(ticker, prices, news_items)
st.write(insights)

# === Q&A ===
st.subheader("❓ Ask a question")
q = st.text_input("e.g., What are key risks noted in the latest 10-K related to supply chain and FX?")
if st.button("Answer"):
    with st.spinner("Retrieving + generating…"):
        docs = retrieve(ticker, q, k=5)
        answer, used = qa_answer(ticker, q, docs)
        st.write(answer)
        with st.expander("Show sources"):
            for d in used:
                meta = d.get('metadata', {})
                snippet = (d.get('page_content','') or '')[:400].replace("\n"," ")
                st.markdown(f"- **{meta.get('form','?')}** {meta.get('filing_date','?')} — {meta.get('source','?')}  \n    `{snippet}…`")

# === RISK ===
st.divider()
st.subheader("🧭 Risk Assessment")
report = risk_report(prices, benchmark='SPY')
st.json(report, expanded=False)

st.caption("© 2025 • Demo by You — For academic/job assessment use only.")
