"""Fable 5 batch translation: submit | cluster | poll | merge.

Usage:
  translate_batch.py submit <subpath> [<subpath> ...]    # batch by subpath
  translate_batch.py cluster [--thr T] <seed> [<seed>..]  # dry-run a t-SNE cluster
  translate_batch.py submit-cluster [--thr T] <seed>..    # batch a t-SNE cluster
  translate_batch.py submit-all [--thr T]                 # one batch per cluster (all remaining)
  translate_batch.py poll-all | merge-all                 # over the batch manifest
  translate_batch.py poll <batch_id>                      # check status
  translate_batch.py merge <batch_id>                     # pull results -> parquet

A <seed> is a subpath leaf (e.g. patthana1) or uid; the cluster is the whole
single-linkage t-SNE blob it sits in (--thr sets the cut, default 4.2 = coarse).
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
                           cluster_assignment, cluster_uids, doc_chunks,
                           load_env, schema, untranslated_docs)

EFFORT = "medium"
OUT = A / "translations_fable.parquet"
MANIFEST = A / "batch_manifest.json"


def build_requests(subpaths=None, uids=None):
    segs = pl.read_parquet(A / "segments.parquet")
    docs = untranslated_docs(subpaths, uids)
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
                    model=MODEL, max_tokens=32000,   # headroom for dense paṭṭhāna/kn chunks
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_prompt(uid, chunk)}],
                    output_config={"effort": EFFORT, "format": schema(seg_ids)},
                ),
            ))
    return reqs, len(docs)


def _cost_estimate(uids):
    """Untranslated segment count + Pali chars + est. batched cost for these uids."""
    segs = pl.read_parquet(A / "segments.parquet")
    done = set(pl.read_parquet(OUT)["segment_id"].to_list()) if OUT.exists() else set()
    sub = segs.filter(pl.col("uid").is_in(uids) & pl.col("pali").is_not_null()
                      & (pl.col("pali").str.strip_chars() != "")
                      & ~pl.col("segment_id").is_in(list(done)))
    chars = sub.select(pl.col("pali").str.len_chars().sum()).item() or 0
    return sub.height, chars, chars / 1000 * 0.021


def cmd_cluster(seeds, thr):
    uids = cluster_uids(seeds, thr)
    docs = untranslated_docs(uids=uids)
    from collections import Counter
    mp = {p["uid"]: p["subpath"] for p in json.loads((A / "map.json").read_text())}
    comp = Counter(mp.get(u, "?") for u in uids)
    nseg, chars, cost = _cost_estimate(docs["uid"].to_list())
    print(f"seeds {seeds} @ thr {thr}: {len(uids)} texts in cluster, "
          f"{len(docs)} untranslated")
    print("composition:", ", ".join(f"{s} ({n})" for s, n in comp.most_common(8)))
    print(f"work: {nseg:,} segments, {chars:,} Pali chars, est. ${cost:.2f} batched")
    return docs


def cmd_submit(subpaths=None, uids=None, desc=""):
    load_env()
    client = anthropic.Anthropic()
    reqs, ndocs = build_requests(subpaths, uids)
    print(f"{ndocs} docs -> {len(reqs)} requests {desc}")
    batch = client.messages.batches.create(requests=reqs)
    print(f"batch id: {batch.id}\nstatus: {batch.processing_status}")
    (A / "last_batch.txt").write_text(batch.id)


def cmd_submit_all(thr=4.2):
    """One batch per coarse t-SNE cluster over all untranslated docs; off-map
    docs go in a single extra batch. Records a manifest for poll-all/merge-all."""
    from collections import defaultdict
    load_env()
    client = anthropic.Anthropic()
    assign = cluster_assignment(thr)
    docs = untranslated_docs()
    groups = defaultdict(list)
    for u in docs["uid"].to_list():
        groups[assign.get(u, -1)].append(u)
    manifest = []
    for cl, us in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        reqs, nd = build_requests(uids=us)
        if not reqs:
            continue
        batch = client.messages.batches.create(requests=reqs)
        _, chars, cost = _cost_estimate(us)
        rec = {"cluster": cl, "batch_id": batch.id, "ndocs": nd,
               "nreq": len(reqs), "est_cost": round(cost, 2)}
        manifest.append(rec)
        MANIFEST.write_text(json.dumps(manifest, indent=1))   # persist incrementally
        tag = "off-map" if cl == -1 else f"cluster {cl}"
        print(f"{tag}: {nd} docs / {len(reqs)} reqs -> {batch.id} (~${cost:.2f})")
    print(f"\nsubmitted {len(manifest)} batches -> {MANIFEST}")


def cmd_poll_all():
    load_env()
    client = anthropic.Anthropic()
    man = json.loads(MANIFEST.read_text())
    ended = 0
    for rec in man:
        b = client.messages.batches.retrieve(rec["batch_id"])
        c = b.request_counts
        done = b.processing_status == "ended"
        ended += done
        tag = "off-map" if rec["cluster"] == -1 else f"cl {rec['cluster']}"
        print(f"{tag:>10} {rec['batch_id']}: {b.processing_status:>10} "
              f"succ={c.succeeded} err={c.errored} proc={c.processing}")
    print(f"\n{ended}/{len(man)} batches ended")


def cmd_merge_all():
    man = json.loads(MANIFEST.read_text())
    for rec in man:
        print(f"-- {rec['batch_id']} (cluster {rec['cluster']}) --")
        cmd_merge(rec["batch_id"])


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


def _parse_thr(rest):
    thr = 4.2
    if "--thr" in rest:
        i = rest.index("--thr")
        thr = float(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    return rest, thr


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "submit":
        cmd_submit(subpaths=rest, desc=f"across {rest}")
    elif cmd == "cluster":
        seeds, thr = _parse_thr(rest)
        cmd_cluster(seeds, thr)
    elif cmd == "submit-cluster":
        seeds, thr = _parse_thr(rest)
        docs = cmd_cluster(seeds, thr)
        cmd_submit(uids=docs["uid"].to_list(),
                   desc=f"for cluster {seeds} @ thr {thr}")
    elif cmd == "submit-all":
        _, thr = _parse_thr(rest)
        cmd_submit_all(thr)
    elif cmd == "poll-all":
        cmd_poll_all()
    elif cmd == "merge-all":
        cmd_merge_all()
    elif cmd == "poll":
        cmd_poll(rest[0])
    elif cmd == "merge":
        cmd_merge(rest[0])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
