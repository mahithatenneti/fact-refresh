import os
from typing import Dict
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SUMM_NAME = os.getenv("SUMM_NAME", "sshleifer/distilbart-cnn-12-6")
_tok = None
_mod = None

def _load_sum():
    global _tok, _mod
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(SUMM_NAME)
        _mod = AutoModelForSeq2SeqLM.from_pretrained(SUMM_NAME)
        _mod.eval()

def summarize(text: str, max_new_tokens: int = 180) -> str:
    _load_sum()
    if not text:
        return ""
    toks = _tok(text, return_tensors="pt", truncation=True, max_length=1024)
    out = _mod.generate(**toks, max_new_tokens=max_new_tokens, num_beams=4)
    return _tok.decode(out[0], skip_special_tokens=True)

def normalize_record(rec: Dict) -> Dict:
    # Ensure required fields and add 'summary'.
    # TODO: Add translation here if needed (OPUS-MT: Helsinki-NLP).
    rec = dict(rec)
    lang = rec.get("lang")
    if not lang:
        lang = detect((rec.get("text") or "")[:500] or "en")
    rec["lang"] = lang or "en"
    if os.getenv("SKIP_SUMMARY", "0") == "1":
        rec["summary"] = (rec.get("text") or "")[:500]
    else:
        rec["summary"] = summarize(rec.get("text",""))
    return rec
