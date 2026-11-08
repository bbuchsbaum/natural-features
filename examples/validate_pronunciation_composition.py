"""Held-out check of opt-in pronunciation composition against the bundled CMUdict.

Each entry is rebuilt from the other entries; its own entry is never consulted.
Usage: python examples/validate_pronunciation_composition.py summary.json
"""

import json
import re
import sys
from collections import Counter
from natural_features.features.speech.pronunciation import load_cmudict
from natural_features.features.speech.pronunciation_compose import compose_pronunciation


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def nostress(p):
    return tuple(x.rstrip("012") for x in p)


d = load_cmudict().entries
rule = re.compile(r"^(.*?)(?:\+|-)")
stats = Counter()
per_rule = {}
errs_num = errs_den = 0
examples = []
words = [w for w in d if re.fullmatch(r"[a-z']+(?:-[a-z']+)*", w)]
for w in words:
    alts, labels = compose_pronunciation(w, d)
    stats["words"] += 1
    if not alts:
        continue
    stats["composed"] += 1
    unique = len(alts) == 1
    stats["unique"] += unique
    first = alts[0]
    ref = d[w]
    exact = first in ref
    exact_ns = nostress(first) in {nostress(r) for r in ref}
    kind = labels[0].split(":", 1)[1]
    kind = (
        "'s"
        if "'s" in kind
        else (
            "compound"
            if kind.count("+")
            and not re.search(
                r"\+(s|es|ed|ing|er|est|y|ly|ness|less|ment|ful|able|ible|ish)$", kind
            )
            else kind.rsplit("+", 1)[-1]
            if "+" in kind
            else "hyphen"
        )
    )
    r = per_rule.setdefault(kind, Counter())
    r["n"] += 1
    r["exact"] += exact
    r["exact_nostress"] += exact_ns
    r["unique"] += unique
    best = min(ref, key=lambda x: lev(nostress(first), nostress(x)))
    errs_num += lev(nostress(first), nostress(best))
    errs_den += len(best)
    stats["exact"] += exact
    stats["exact_nostress"] += exact_ns
    if not exact_ns and len(examples) < 40 and unique:
        examples.append((w, labels[0], " ".join(first), " ".join(ref[0])))
out = {
    "stats": dict(stats),
    "phone_error_rate_nostress": errs_num / errs_den,
    "per_rule": {
        k: dict(v) for k, v in sorted(per_rule.items(), key=lambda kv: -kv[1]["n"])
    },
    "examples_wrong": examples,
}
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: out[k] for k in ("stats", "phone_error_rate_nostress")}, indent=1))
for k, v in out["per_rule"].items():
    print(
        f"{k:10s} n={v['n']:6d} exact={v['exact'] / v['n']:.3f} nostress={v['exact_nostress'] / v['n']:.3f} unique={v['unique'] / v['n']:.3f}"
    )
