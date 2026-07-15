"""Measure real Fable 5 translation cost on a few docs (synchronous).

Translates a handful of untranslated verse docs, prints per-doc token usage and
batched-equivalent cost, and saves the translations to
artifacts/fable_probe.json so the spend isn't wasted.
"""

import json
import sys

import anthropic
import polars as pl

from translate_lib import (A, MODEL, PRICE_IN, PRICE_OUT, SYSTEM, build_prompt,
                           doc_chunks, load_env, schema)

PROBE_UIDS = ["bv1", "bv2", "vv1", "pv1", "thi-ap1", "pv21"]
EFFORT = "medium"


def main() -> None:
    load_env()
    client = anthropic.Anthropic()
    segs = pl.read_parquet(A / "segments.parquet")

    results, tin, tout = {}, 0, 0
    for uid in PROBE_UIDS:
        for cid, chunk in doc_chunks(uid, segs):
            seg_ids = [s for s, _ in chunk]
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_prompt(uid, chunk)}],
                    output_config={"effort": EFFORT, "format": schema(seg_ids)},
                )
            except anthropic.BadRequestError as e:
                sys.exit(f"BadRequest ({uid}): {e.message}\n"
                         "(if this mentions data retention, the org is ZDR and "
                         "Fable 5 is unavailable — we'd fall back to Opus 4.8)")
            if resp.stop_reason == "refusal":
                print(f"  {cid}: REFUSAL — skipping")
                continue
            text = next(b.text for b in resp.content if b.type == "text")
            data = json.loads(text)
            for t in data["translations"]:
                results[t["id"]] = t["en"]
            u = resp.usage
            tin += u.input_tokens
            tout += u.output_tokens
            print(f"  {cid}: {len(chunk)} segs · in {u.input_tokens} · "
                  f"out {u.output_tokens}")

    (A / "fable_probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    pali_chars = int(
        segs.filter(pl.col("uid").is_in(PROBE_UIDS) & pl.col("pali").is_not_null())
        .select(pl.col("pali").str.len_chars().sum()).item()
    )
    cost = tin * PRICE_IN + tout * PRICE_OUT      # batched-equivalent
    print(f"\n{len(results)} segments · {pali_chars} Pali chars")
    print(f"tokens: in {tin} · out {tout}")
    print(f"batched-equiv cost: ${cost:.4f}  "
          f"(synchronous actual: ${tin*PRICE_IN*2 + tout*PRICE_OUT*2:.4f})")
    print(f"per 1k Pali chars (batched): ${cost/pali_chars*1000:.4f}")
    print(f"-> full 12.67M-char corpus extrapolates to "
          f"${cost/pali_chars*12_666_784:.0f} batched")


if __name__ == "__main__":
    main()
