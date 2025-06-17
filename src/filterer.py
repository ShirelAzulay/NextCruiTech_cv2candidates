# src/filterer.py
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI(api_key="sk-proj-...")

def filter_by_prompt(resumes: dict, prompt: str) -> dict:
    filtered = {}
    for cid, text in resumes.items():
        full_prompt = (
            f"{prompt}\n\n"
            f"Resume Text (first 2000 chars):\n{text[:2000]}\n\n"
            "Answer yes or no."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content": full_prompt}],
            temperature=0
        )
        ans = resp.choices[0].message.content.strip().lower()
        logger.debug(f"Filter '{cid}': {ans}")
        if ans.startswith("yes"):
            filtered[cid] = text
    return filtered
