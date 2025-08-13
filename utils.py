import os, re, math, io, hashlib, datetime as dt
import numpy as np
import pandas as pd

def daterange_str(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
