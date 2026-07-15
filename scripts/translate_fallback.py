"""GPT-5.6 sol fallback for segments Fable refused/failed to complete.

Fable's refusals are the next-best translator's job (user policy): a different
vendor won't share Fable's refusal boundary, and smaller chunks dodge the dense-
text truncation that also fails some paṭṭhāna/niddesa passages. Translates via
OpenRouter (openai/gpt-5.6-sol), writing artifacts/translations_gpt.parquet
(segment_id, english); merge_translations.py then folds it in as gpt-5.6-sol.

Usage: translate_fallback.py            # translate all still-refused segments
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import polars as pl

from translate_lib import SYSTEM, build_prompt, load_env

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts"
OUT = A / "translations_gpt.parquet"
MODEL = "openai/gpt-5.6-sol"
URL = "https://openrouter.ai/api/v1/chat/completions"
CAP = 40          # smaller than Fable's 100 — dense chunks were truncating
CONC = 5


def or_schema() -> dict:
    return {"type": "json_schema", "json_schema": {"name": "translations",
        "strict": True, "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["translations"],
            "properties": {"translations": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "en"],
                "properties": {"id": {"type": "string"}, "en": {"type": "string"}}}}}}}}


def refused_chunks():
    """[(uid, [(segment_id, pali), ...]), ...] over still-untranslated machine segments."""
    d = pl.read_parquet(A / "documents.parquet")
    s = pl.read_parquet(A / "segments.parquet")
    done = set(pl.read_parquet(OUT)["segment_id"].to_list()) if OUT.exists() else set()
    mach = set(d.filter((pl.col("translator") == "claude-fable-5")
                        | (pl.col("translated_frac") < 0.05))["uid"].to_list())
    ref = (s.filter(pl.col("uid").is_in(list(mach)) & pl.col("english").is_null()
                    & pl.col("pali").is_not_null() & (pl.col("pali").str.strip_chars() != ""))
           .sort(["uid", "seq"]))
    out = []
    for (uid,), df in ref.group_by("uid", maintain_order=True):
        rows = [(sid, p) for sid, p in df.select("segment_id", "pali").rows()
                if sid not in done]
        for i in range(0, len(rows), CAP):
            out.append((uid, rows[i:i + CAP]))
    return out


async def translate(client, uid, chunk, sem, i, total):
    seg_ids = [s for s, _ in chunk]
    body = {"model": MODEL, "max_tokens": 16000,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": build_prompt(uid, chunk)}],
            "response_format": or_schema()}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(URL, json=body, timeout=600)
                if r.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                d = r.json()
                if "error" in d:
                    raise RuntimeError(d["error"])
                data = json.loads(d["choices"][0]["message"]["content"])
                rows = [{"segment_id": t["id"], "english": t["en"]}
                        for t in data["translations"] if t.get("en")]
                print(f"[{i}/{total}] {uid}: {len(rows)}/{len(seg_ids)} segs")
                return rows
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError, KeyError) as e:
                err = str(e)[:160]
                await asyncio.sleep(8 * (attempt + 1))
        print(f"[{i}/{total}] FAIL {uid}: {err}")
        return []


async def main() -> None:
    load_env()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY not in .env")
    chunks = refused_chunks()
    if not chunks:
        print("nothing to fall back on")
        return
    total = len(chunks)
    print(f"{sum(len(c) for _, c in chunks)} segments in {total} chunks -> {MODEL}")
    sem = asyncio.Semaphore(CONC)
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers) as client:
        results = await asyncio.gather(*[
            translate(client, uid, chunk, sem, i + 1, total)
            for i, (uid, chunk) in enumerate(chunks)])
    rows = [r for chunk in results for r in chunk]
    new = pl.DataFrame(rows, schema={"segment_id": pl.Utf8, "english": pl.Utf8})
    if OUT.exists():
        new = pl.concat([pl.read_parquet(OUT), new]).unique("segment_id", keep="last")
    new.write_parquet(OUT)
    print(f"wrote {len(rows)} segments; {OUT} now holds {len(new)}")


if __name__ == "__main__":
    asyncio.run(main())
