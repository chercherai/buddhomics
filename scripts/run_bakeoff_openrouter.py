"""Bake-off round 2: GPT-5.6 family via OpenRouter, same passages/prompt/schema.

Writes artifacts/bakeoff_results_gpt.json in the same shape as bakeoff_results.json.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent

for line in (REPO / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

MODELS = ["openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-5.6-luna"]
URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """You are translating Pali Buddhist canonical texts into English.

You will receive numbered segments from a single continuous passage. Translate each
segment into clear, accurate English, in the register of a good modern scholarly
translation (like Bhikkhu Sujato's or Bhikkhu Bodhi's). Preserve doctrinal
terminology precisely; use established renderings where they exist. Translate
segment by segment: each segment's translation must correspond to that segment's
Pali, not merged or redistributed across segments. Keep verse compact and natural.
If a segment is a fragment (verse quarter or clause), translate the fragment so the
sequence reads continuously."""


def schema(seg_ids):
    return {
        "name": "translations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "enum": seg_ids},
                            "en": {"type": "string"},
                        },
                        "required": ["id", "en"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    }


async def translate(client, model, passage, sem):
    seg_ids = [s["id"] for s in passage["segments"]]
    text = "\n".join(f"{s['id']}\t{s['pali']}" for s in passage["segments"])
    prompt = (
        f"Passage from {passage['label']} ({passage['uid']}).\n"
        f"Translate all {len(seg_ids)} segments.\n\n{text}"
    )
    body = {
        "model": model,
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema(seg_ids)},
    }
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
                content = d["choices"][0]["message"]["content"]
                data = json.loads(content)
                segs = {t["id"]: t["en"] for t in data["translations"]}
                u = d.get("usage", {})
                usage = dict(input=u.get("prompt_tokens", 0), output=u.get("completion_tokens", 0))
                print(f"done {passage['uid']:14} {model:26} ({usage['output']} out tok)")
                return passage["uid"], model, {"segments": segs, "usage": usage}
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError, KeyError) as e:
                err = str(e)[:200]
                await asyncio.sleep(8 * (attempt + 1))
        print(f"FAIL {passage['uid']} {model}: {err}")
        return passage["uid"], model, {"error": err}


async def main():
    passages = json.loads((REPO / "artifacts" / "bakeoff_passages.json").read_text())
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [translate(client, m, p, sem) for p in passages for m in MODELS]
        results = {}
        for uid, model, res in await asyncio.gather(*tasks):
            results.setdefault(uid, {})[model] = res
    out = REPO / "artifacts" / "bakeoff_results_gpt.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
