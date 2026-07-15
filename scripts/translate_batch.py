"""Fable 5 batch translation: submit | poll | merge.

Usage:
  translate_batch.py submit <subpath> [<subpath> ...]   # create + submit a batch
  translate_batch.py poll <batch_id>                     # check status
  translate_batch.py merge <batch_id>                    # pull results -> parquet

Writes artifacts/translations_fable.parquet (segment_id, english) accumulating
across batches; run rebuild afterwards to fold into the substrate.
"""

import json
import sys
from pathlib import Path

import anthropic
import polars as pl
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from translate_lib import (A, MODEL, PRICE_IN, PRICE_OUT, SYSTEM, build_prompt,
                           doc_chunks, load_env, schema, untranslated_docs)

EFFORT = "medium"
OUT = A / "translations_fable.parquet"


def build_requests(subpaths):
    segs = pl.read_parquet(A / "segments.parquet")
    docs = untranslated_docs(subpaths)
    done = set()
    if OUT.exists():
        done = set(pl.read_parquet(OUT)["segment_id"].to_list())
    reqs = []
    for uid in docs["uid"].to_list():
        for cid, chunk in doc_chunks(uid, segs):
            chunk = [(s, p) for s, p in chunk if s not in done]
            if not chunk:
                continue
            seg_ids = [s for s, _ in chunk]
            reqs.append(Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL, max_tokens=16000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_prompt(uid, chunk)}],
                    output_config={"effort": EFFORT, "format": schema(seg_ids)},
                ),
            ))
    return reqs, len(docs)


def cmd_submit(subpaths):
    load_env()
    client = anthropic.Anthropic()
    reqs, ndocs = build_requests(subpaths)
    print(f"{ndocs} docs -> {len(reqs)} requests across {subpaths}")
    batch = client.messages.batches.create(requests=reqs)
    print(f"batch id: {batch.id}\nstatus: {batch.processing_status}")
    (A / "last_batch.txt").write_text(batch.id)


def cmd_poll(batch_id):
    load_env()
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(batch_id)
    print(f"status: {b.processing_status}")
    print(f"counts: {b.request_counts}")


def cmd_merge(batch_id):
    load_env()
    client = anthropic.Anthropic()
    rows, tin, tout, errs, refusals = [], 0, 0, 0, 0
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            errs += 1
            continue
        msg = res.result.message
        if msg.stop_reason == "refusal":
            refusals += 1
            continue
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        try:
            text = next(b.text for b in msg.content if b.type == "text")
            for t in json.loads(text)["translations"]:
                rows.append({"segment_id": t["id"], "english": t["en"]})
        except (StopIteration, json.JSONDecodeError, KeyError):
            errs += 1
    new = pl.DataFrame(rows, schema={"segment_id": pl.Utf8, "english": pl.Utf8})
    if OUT.exists():
        new = pl.concat([pl.read_parquet(OUT), new]).unique("segment_id", keep="last")
    new.write_parquet(OUT)
    cost = tin * PRICE_IN + tout * PRICE_OUT
    print(f"merged {len(rows)} segments ({errs} errors, {refusals} refusals)")
    print(f"tokens in {tin} / out {tout} · batched cost ${cost:.2f}")
    print(f"{OUT} now holds {len(new)} segment translations")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "submit":
        cmd_submit(rest)
    elif cmd == "poll":
        cmd_poll(rest[0])
    elif cmd == "merge":
        cmd_merge(rest[0])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
