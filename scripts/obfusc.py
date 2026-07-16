"""Reversible obfuscation for served commentary text (keeps it out of search
indexes / casual scraping). NOT encryption — the key is derivable client-side;
it reduces exposure, it is not a license. Must match the JS decoder in the site.
"""
import base64

SALT = "buddhomics-comm-v1"


def _fnv(s: str) -> int:
    x = 2166136261
    for b in s.encode("utf-8"):
        x = ((x ^ b) * 16777619) & 0xFFFFFFFF
    return x


def encode(text: str, sid: str) -> str:
    data = text.encode("utf-8")
    s = _fnv(SALT + sid) or 1
    out = bytearray(len(data))
    for i, b in enumerate(data):
        s = (s ^ ((s << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
        s = (s ^ (s >> 17)) & 0xFFFFFFFF
        s = (s ^ ((s << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
        out[i] = b ^ (s & 0xFF)
    return base64.b64encode(bytes(out)).decode("ascii")


def decode(b64: str, sid: str) -> str:
    data = bytearray(base64.b64decode(b64))
    s = _fnv(SALT + sid) or 1
    for i in range(len(data)):
        s = (s ^ ((s << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
        s = (s ^ (s >> 17)) & 0xFFFFFFFF
        s = (s ^ ((s << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
        data[i] ^= s & 0xFF
    return bytes(data).decode("utf-8")
