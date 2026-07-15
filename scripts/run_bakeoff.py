"""Translation bake-off: run the 12 passages through Opus 4.8, Sonnet 5, Haiku 4.5.

Reads ANTHROPIC_API_KEY from .env. Writes artifacts/bakeoff_results.json:
{passage_uid: {model: {segments: {id: en}, usage: {...}}}}
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic

REPO = Path(__file__).resolve().parent.parent

# load .env
for line in (REPO / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

MODELS = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]

SYSTEM = """You are translating Pali Buddhist canonical texts into English.

You will receive numbered segments from a single continuous passage. Translate each
segment into clear, accurate English, in the register of a good modern scholarly
translation (like Bhikkhu Sujato's or Bhikkhu Bodhi's). Preserve doctrinal
terminology precisely; use established renderings where they exist. Translate
segment by segment: each segment's translation must correspond to that segment's
Pali, not merged or redistributed across segments. Keep verse compact and natural.
If a segment is a fragment (verse quarter or clause), translate the fragment so the
sequence reads continuously."""


def schema(seg_ids: list[str]) -> dict:
    return {
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
    }


async def translate(client, model: str, passage: dict, sem) -> tuple[str, str, dict]:
    seg_ids = [s["id"] for s in passage["segments"]]
    text = "\n".join(f"{s['id']}\t{s['pali']}" for s in passage["segments"])
    prompt = (
        f"Passage from {passage['label']} ({passage['uid']}).\n"
        f"Translate all {len(seg_ids)} segments.\n\n{text}"
    )
    kwargs = dict(
        model=model,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": schema(seg_ids)}},
    )
    if model in ("claude-opus-4-8", "claude-sonnet-5"):
        kwargs["thinking"] = {"type": "adaptive"}
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.messages.create(**kwargs)
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError):
                await asyncio.sleep(15 * (attempt + 1))
        else:
            return passage["uid"], model, {"error": "retries exhausted"}
    if resp.stop_reason == "refusal":
        return passage["uid"], model, {"error": "refusal"}
    raw = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(raw)
    segs = {t["id"]: t["en"] for t in data["translations"]}
    usage = dict(input=resp.usage.input_tokens, output=resp.usage.output_tokens)
    print(f"done {passage['uid']:14} {model:20} ({usage['output']} out tok)")
    return passage["uid"], model, {"segments": segs, "usage": usage}


async def main() -> None:
    passages = json.loads((REPO / "artifacts" / "bakeoff_passages.json").read_text())
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(6)
    tasks = [translate(client, m, p, sem) for p in passages for m in MODELS]
    results: dict = {}
    for uid, model, res in await asyncio.gather(*tasks):
        results.setdefault(uid, {})[model] = res
    out = REPO / "artifacts" / "bakeoff_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
