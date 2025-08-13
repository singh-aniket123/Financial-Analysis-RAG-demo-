import os, textwrap, datetime as dt
from typing import List, Dict, Any
import pandas as pd
from .timeseries import features, features_summary
from .news import fetch_news_rss

def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _llm_chat(prompt: str, system: str = "You are a cautious financial analyst. Never give investment advice; provide balanced, sourced analysis.") -> str:
    client = _openai_client()
    model = os.getenv("LLM_MODEL","gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e:
        # fallback shorter message if quota/model issue
        return f"(LLM error: {e})"

def _ts_blurb(ticker: str, prices: pd.DataFrame) -> str:
    f = features(prices)
    last = f.iloc[-1]
    ytd = (prices['Close'].iloc[-1]/prices['Close'].iloc[0]-1) if len(prices)>1 else 0
    return textwrap.dedent(f\"\"\"
        Ticker: {ticker}
        Period: {prices.index.min().date()} → {prices.index.max().date()}
        Last Close: {prices['Close'].iloc[-1]:.2f}
        YTD Return (period): {ytd:.2%}
        Ann Vol (20d): {last.get('vol_20', float('nan')):.2%}
        Drawdown: {last.get('drawdown', float('nan')):.2%}
        Beta(60d vs SPY): {last.get('beta_60', float('nan')):.2f}
        MA20{'>' if last.get('ma_20',0)>last.get('ma_50',0) else '<='}MA50
    \"\"\")

def generate_insights(ticker: str, prices: pd.DataFrame, news_items: List[Dict[str,str]]) -> str:
    ts = _ts_blurb(ticker, prices)
    headlines = "\\n".join([f"- {n['title']} ({n['published']})" for n in news_items[:5]])
    prompt = f\"\"\"
    Using the time-series snapshot and recent headlines, write a concise, neutral insight report (5-8 bullets).
    Highlight trends (momentum, volatility regime, drawdown), macro/firm-specific catalysts from headlines,
    and connect risks to the current technical state. Do not recommend buy/sell.

    Time-series:
    {ts}

    Headlines:
    {headlines}
    \"\"\"
    return _llm_chat(prompt)

def qa_answer(ticker: str, question: str, docs: List[Dict[str,Any]]):
    # Build citations and context
    context = "\\n\\n".join([f"[{i+1}] {d['page_content']}" for i,d in enumerate(docs)])
    sources = "\\n".join([f"[{i+1}] {d['metadata'].get('source','?')} ({d['metadata'].get('filing_date','?')})" for i,d in enumerate(docs)])
    prompt = f\"\"\"
    Answer the user's question strictly using the provided SEC filing excerpts.
    Quote exact phrases sparingly and provide a short, sourced answer with bracketed citations like [1], [2].
    If unsure, say so.

    Question: {question}

    Context:
    {context}

    When relevant, include dates and form type.
    Sources:
    {sources}
    \"\"\"
    ans = _llm_chat(prompt)
    return ans, docs
