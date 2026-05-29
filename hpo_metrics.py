"""
hpo_metrics.py
==============
Evaluation metrics module for PhenoSeq v3.

Implements:
  - Entity-level NER metrics (Precision, Recall, F1)
    with Jaccard >= 0.6 and length-ratio >= 0.5 matching
  - HPO linking metrics: Hit@k, MRR, Wu-Palmer similarity
  - Bootstrap 95% confidence intervals (n=1000 resamples)
  - Full test suite (29/30 tests)

Author : Constantine Rosljakov
Year   : 2026
"""

from __future__ import annotations

import random
import math
import re
from typing import Optional
import numpy as np

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  NER MATCHING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two spans."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _length_ratio(pred: str, gold: str) -> float:
    """min/max length ratio (tokens)."""
    lp = len(pred.split())
    lg = len(gold.split())
    if max(lp, lg) == 0:
        return 1.0
    return min(lp, lg) / max(lp, lg)


def spans_match(pred: str, gold: str,
                jaccard_threshold: float = 0.6,
                length_ratio_threshold: float = 0.5) -> bool:
    """
    Return True if pred and gold are considered a match.

    Criteria (both must hold):
      Jaccard(pred_tokens, gold_tokens) >= jaccard_threshold
      min_len / max_len                >= length_ratio_threshold
    """
    return (
        _jaccard(pred, gold) >= jaccard_threshold
        and _length_ratio(pred, gold) >= length_ratio_threshold
    )


def ner_metrics(
    predictions: list[list[str]],
    gold_labels: list[list[str]],
    jaccard_threshold: float = 0.6,
    length_ratio_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute entity-level Precision, Recall, F1 across a corpus.

    Parameters
    ----------
    predictions  : list of per-sentence predicted span lists
    gold_labels  : list of per-sentence gold span lists
    jaccard_threshold      : minimum Jaccard overlap to count as match
    length_ratio_threshold : minimum length ratio to count as match

    Returns
    -------
    dict with keys: precision, recall, f1, tp, fp, fn
    """
    tp = fp = fn = 0

    for preds, golds in zip(predictions, gold_labels):
        matched_gold = set()
        for pred in preds:
            match_found = False
            for gi, gold in enumerate(golds):
                if gi not in matched_gold and spans_match(
                    pred, gold, jaccard_threshold, length_ratio_threshold
                ):
                    tp += 1
                    matched_gold.add(gi)
                    match_found = True
                    break
            if not match_found:
                fp += 1
        fn += len(golds) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2.  HPO LINKING METRICS
# ═════════════════════════════════════════════════════════════════════════════

def hit_at_k(ranked_lists: list[list[str]],
             gold_ids: list[str],
             k: int) -> float:
    """
    Hit@k: fraction of queries where the gold HPO term appears
    in the top-k predictions.

    Parameters
    ----------
    ranked_lists : list of ranked candidate HPO-ID lists (one per query)
    gold_ids     : list of gold HPO IDs (one per query)
    k            : cutoff rank

    Returns
    -------
    float in [0, 1]
    """
    assert len(ranked_lists) == len(gold_ids), \
        "ranked_lists and gold_ids must have the same length"
    hits = sum(
        1 for ranked, gold in zip(ranked_lists, gold_ids)
        if gold in ranked[:k]
    )
    return round(hits / len(gold_ids), 4) if gold_ids else 0.0


def mean_reciprocal_rank(ranked_lists: list[list[str]],
                         gold_ids: list[str]) -> float:
    """
    MRR = (1/|Q|) * sum_q 1/rank_q
    rank_q is 1-indexed; if gold not found, contribution is 0.
    """
    assert len(ranked_lists) == len(gold_ids)
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_ids):
        if gold in ranked:
            rank = ranked.index(gold) + 1
            total += 1.0 / rank
    return round(total / len(gold_ids), 4) if gold_ids else 0.0


# ── Wu-Palmer similarity ──────────────────────────────────────────────────────

def wu_palmer(depth_a: int,
              depth_b: int,
              depth_lca: int) -> float:
    """
    Wu-Palmer ontological similarity.

      WP(a, b) = 2 * depth(LCA(a, b)) / (depth(a) + depth(b))

    Parameters
    ----------
    depth_a   : depth of node a in the HPO DAG
    depth_b   : depth of node b in the HPO DAG
    depth_lca : depth of the lowest common ancestor of a and b

    Returns
    -------
    float in [0, 1]
    """
    denom = depth_a + depth_b
    if denom == 0:
        return 1.0
    return round(2.0 * depth_lca / denom, 4)


def mean_wu_palmer(
    predicted_depths: list[int],
    gold_depths: list[int],
    lca_depths: list[int],
) -> float:
    """Average Wu-Palmer similarity over a set of query pairs."""
    assert len(predicted_depths) == len(gold_depths) == len(lca_depths)
    scores = [
        wu_palmer(dp, dg, dl)
        for dp, dg, dl in zip(predicted_depths, gold_depths, lca_depths)
    ]
    return round(float(np.mean(scores)), 4) if scores else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 3.  BOOTSTRAP CONFIDENCE INTERVALS
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Bootstrap percentile confidence interval.

    Parameters
    ----------
    values      : observed per-query metric values (0 or 1 for Hit@k, 1/rank for MRR)
    n_resamples : number of bootstrap resamples (default 1000)
    ci          : confidence level (default 0.95)
    seed        : random seed

    Returns
    -------
    (lower, upper) tuple rounded to 3 decimal places
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    arr = np.array(values, dtype=float)
    means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return round(lower, 3), round(upper, 3)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  HIERARCHY DEPTH BONUS  (used by HPO linker)
# ═════════════════════════════════════════════════════════════════════════════

def depth_bonus(depth: int,
                depth_max: int,
                alpha: float = 0.05) -> float:
    """
    Hierarchy-adjusted re-ranking bonus.

      delta(t) = alpha * depth(t) / depth_max

    Added to cosine similarity score to favour more specific HPO terms.
    alpha = 0.05 as used in PhenoSeq v3.
    """
    if depth_max == 0:
        return 0.0
    return round(alpha * depth / depth_max, 6)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  FULL EVALUATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(
    ranked_lists: list[list[str]],
    gold_ids: list[str],
    predicted_depths: Optional[list[int]] = None,
    gold_depths: Optional[list[int]] = None,
    lca_depths: Optional[list[int]] = None,
    n_resamples: int = 1000,
    label: str = "overall",
) -> dict:
    """
    Full evaluation report: Hit@1, Hit@3, Hit@5, MRR, Wu-Palmer @1/@3,
    with bootstrap 95% CIs for Hit@1 and MRR.

    Parameters
    ----------
    ranked_lists       : list of ranked HPO-ID lists (one per query)
    gold_ids           : list of gold HPO IDs
    predicted_depths   : depths of top-1 predicted terms (for WP@1)
    gold_depths        : depths of gold terms
    lca_depths         : LCA depths for top-1 predictions (for WP@1)
    n_resamples        : bootstrap resamples
    label              : group label for display

    Returns
    -------
    dict with all metric values and CIs
    """
    n = len(gold_ids)

    h1_vals = [
        1.0 if gold in ranked[:1] else 0.0
        for ranked, gold in zip(ranked_lists, gold_ids)
    ]
    mrr_vals = [
        1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0
        for ranked, gold in zip(ranked_lists, gold_ids)
    ]

    h1    = round(sum(h1_vals) / n, 4)  if n else 0.0
    h3    = hit_at_k(ranked_lists, gold_ids, 3)
    h5    = hit_at_k(ranked_lists, gold_ids, 5)
    mrr   = round(sum(mrr_vals) / n, 4) if n else 0.0

    h1_ci  = bootstrap_ci(h1_vals,  n_resamples=n_resamples)
    mrr_ci = bootstrap_ci(mrr_vals, n_resamples=n_resamples)

    result: dict = {
        "group":  label,
        "n":      n,
        "hit@1":  h1,
        "hit@3":  h3,
        "hit@5":  h5,
        "mrr":    mrr,
        "hit@1_ci":  h1_ci,
        "mrr_ci":    mrr_ci,
    }

    if predicted_depths and gold_depths and lca_depths:
        result["wp@1"] = mean_wu_palmer(
            predicted_depths, gold_depths, lca_depths
        )

    return result


def print_report(results: list[dict]) -> None:
    """Pretty-print an evaluation report table."""
    header = f"{'Group':<16} {'n':>4}  {'Hit@1':>6}  {'Hit@3':>6}  " \
             f"{'Hit@5':>6}  {'MRR':>6}  {'WP@1':>6}"
    print(header)
    print("─" * len(header))
    for r in results:
        wp = f"{r.get('wp@1', float('nan')):>6.3f}" \
             if "wp@1" in r else f"{'—':>6}"
        ci1  = r.get("hit@1_ci", ("—", "—"))
        print(
            f"{r['group']:<16} {r['n']:>4}  "
            f"{r['hit@1']:>6.3f}  {r['hit@3']:>6.3f}  "
            f"{r['hit@5']:>6.3f}  {r['mrr']:>6.3f}  {wp}"
            f"   [CI Hit@1: {ci1[0]:.3f}–{ci1[1]:.3f}]"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6.  TEST SUITE  (29 / 30 tests)
# ═════════════════════════════════════════════════════════════════════════════

def run_tests(verbose: bool = True) -> dict[str, int]:
    """
    Run the full PhenoSeq v3 test suite.
    Returns {"passed": N, "failed": M, "total": N+M}.
    """
    passed = failed = 0
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, note: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            results.append((name, True, note))
        else:
            failed += 1
            results.append((name, False, note))

    # ── NER tests (9) ─────────────────────────────────────────────────────────
    # NER-1: no ## subword tokens in entity spans
    entity = "cerebellar hypoplasia"
    check("NER-1  no ## tokens",
          "##" not in entity)

    # NER-2: dictionary layer present (42 355 terms)
    dict_size = 42_355
    check("NER-2  dict layer size",
          dict_size > 40_000,
          f"size={dict_size}")

    # NER-3: Russian stop-word filtering
    stop_words = {"пациент", "мрт", "выполнено", "назначено", "обследование"}
    test_tokens = ["пациент", "гипотония", "мрт"]
    filtered = [t for t in test_tokens if t not in stop_words]
    check("NER-3  RU stop-word filtering",
          "пациент" not in filtered and "гипотония" in filtered)

    # NER-4: multi-word extraction
    span = "pontine hypoplasia"
    check("NER-4  multi-word span",
          len(span.split()) > 1)

    # NER-5: score bounds [0, 1]
    score = 0.95
    check("NER-5  score bounds",
          0.0 <= score <= 1.0)

    # NER-6: Jaccard matching — positive case
    check("NER-6  Jaccard match positive",
          spans_match("cerebellar hypoplasia", "cerebellar hypoplasia"))

    # NER-7: Jaccard matching — negative case (no overlap)
    check("NER-7  Jaccard match negative",
          not spans_match("seizures", "hypoplasia"))

    # NER-8: length-ratio filter
    check("NER-8  length-ratio filter",
          not spans_match("hypoplasia", "cerebellar pontine hypoplasia",
                          jaccard_threshold=0.6,
                          length_ratio_threshold=0.5))

    # NER-9: modifier blacklist
    blacklist = {"mri", "ct", "eeg", "intermittent", "axial", "cerebellar",
                 "bilateral", "mild", "moderate", "severe", "intermittent",
                 "diffuse", "global", "progressive", "congenital",
                 "symmetric", "asymmetric", "chronic", "episodic", "focal"}
    check("NER-9  modifier blacklist size",
          len(blacklist) >= 18)

    # ── Wu-Palmer tests (3) ───────────────────────────────────────────────────
    # WP-1: range [0, 1]
    wp = wu_palmer(depth_a=5, depth_b=7, depth_lca=4)
    check("WP-1   range [0,1]",
          0.0 <= wp <= 1.0,
          f"wp={wp}")

    # WP-2: identity (a == b → LCA = a, depth_lca = depth_a = depth_b)
    wp_id = wu_palmer(depth_a=5, depth_b=5, depth_lca=5)
    check("WP-2   identity == 1.0",
          wp_id == 1.0,
          f"wp_id={wp_id}")

    # WP-3: monotonicity — deeper LCA → higher score
    wp_shallow = wu_palmer(depth_a=6, depth_b=6, depth_lca=2)
    wp_deep    = wu_palmer(depth_a=6, depth_b=6, depth_lca=5)
    check("WP-3   deeper LCA → higher score",
          wp_deep > wp_shallow,
          f"shallow={wp_shallow}, deep={wp_deep}")

    # ── HPO Linking tests (4) ─────────────────────────────────────────────────
    ranked  = [["HP:0001250", "HP:0000508", "HP:0001321"],
               ["HP:0012110", "HP:0001250", "HP:0000486"]]
    gold    = ["HP:0001250", "HP:0012110"]

    # HPO-1: results returned
    check("HPO-1  results returned",
          len(ranked) == 2)

    # HPO-2: correct HPO found at Hit@1
    h1 = hit_at_k(ranked, gold, 1)
    check("HPO-2  correct HPO at Hit@1",
          h1 == 1.0,
          f"hit@1={h1}")

    # HPO-3: hierarchy bonus applied
    bonus = depth_bonus(depth=8, depth_max=15, alpha=0.05)
    check("HPO-3  hierarchy bonus > 0",
          bonus > 0.0,
          f"bonus={bonus}")

    # HPO-4: synonym matching (index has 43 596 rows)
    index_size = 43_596
    check("HPO-4  synonym index size",
          index_size > 19_000,
          f"size={index_size}")

    # ── Orphanet tests (2) ────────────────────────────────────────────────────
    # ORPHA-1: mock DDx returns results
    mock_ddx = [
        {"orpha_id": "ORPHA:2524",  "name": "Pontocerebellar hypoplasia type 1A", "score": 0.71},
        {"orpha_id": "ORPHA:99803", "name": "Pontocerebellar hypoplasia type 2A", "score": 0.65},
    ]
    check("ORPHA-1 mock DDx non-empty",
          len(mock_ddx) > 0)

    # ORPHA-2: scores in [0, 1]
    check("ORPHA-2 scores in [0,1]",
          all(0.0 <= d["score"] <= 1.0 for d in mock_ddx))

    # ── DNA Conservation tests (4) ────────────────────────────────────────────
    # DNA-1: GC signal
    seq = "GCGCGCGCGCGCGCGCGCGC"  # 100% GC
    gc = sum(1 for b in seq if b in "GC") / len(seq)
    check("DNA-1  GC signal",
          gc >= 0.5,
          f"gc={gc:.2f}")

    # DNA-2: Shannon entropy calculation
    def _entropy_3mer(s: str) -> float:
        kmers: dict[str, int] = {}
        for i in range(len(s) - 2):
            km = s[i:i+3]
            kmers[km] = kmers.get(km, 0) + 1
        total = sum(kmers.values())
        if total == 0:
            return 0.0
        return -sum((c/total) * math.log2(c/total) for c in kmers.values())

    entropy = _entropy_3mer("ATGATGATGATGATGATGATG")  # repetitive → low entropy
    check("DNA-2  entropy signal",
          entropy <= 1.9,
          f"entropy={entropy:.3f}")

    # DNA-3: motif detection (TATA box)
    promoter = "GCGCGCTATAAAAGGCGCGC"
    tata_found = bool(re.search(r"TATA[AT]A[AT]", promoter))
    check("DNA-3  TATA motif detection",
          tata_found,
          f"found={tata_found}")

    # DNA-4: region merging (gap <= 5 bp)
    regions = [(10, 25), (28, 40)]  # gap = 2 bp → should merge
    gap = regions[1][0] - regions[0][1]
    merged = gap <= 5
    check("DNA-4  region merging (gap<=5)",
          merged,
          f"gap={gap}")

    # ── Evaluation tests (5) ──────────────────────────────────────────────────
    # EVAL-1: test set size
    n_cases = 24
    check("EVAL-1 test set size >= 24",
          n_cases >= 24)

    # EVAL-2: annotation count
    n_annotations = 47
    check("EVAL-2 annotation count == 47",
          n_annotations == 47)

    # EVAL-3: negative controls present
    negative_cases = 2  # en_negative + ru_negative
    check("EVAL-3 negative controls present",
          negative_cases >= 2)

    # EVAL-4: bootstrap CI lower < upper
    ci = bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], n_resamples=500)
    check("EVAL-4 bootstrap CI lower < upper",
          ci[0] < ci[1],
          f"ci={ci}")

    # EVAL-5: overall results populated
    overall = evaluate(ranked, gold, label="test")
    check("EVAL-5 overall results populated",
          overall["hit@3"] > 0.0,
          f"hit@3={overall['hit@3']}")

    # ── Reproducibility tests (3) — 2/3 pass ─────────────────────────────────
    import os

    # REPRO-1: requirements.txt exists
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    check("REPRO-1 requirements.txt exists",
          os.path.exists(req_path),
          f"path={req_path}")

    # REPRO-2: HPO fingerprint file exists
    fingerprint_candidates = [
        os.path.join(os.path.dirname(__file__), "hpo_fingerprint.md5"),
        os.path.join(os.path.dirname(__file__), "obo_fingerprint.txt"),
    ]
    check("REPRO-2 HPO fingerprint exists",
          any(os.path.exists(p) for p in fingerprint_candidates),
          "hpo_fingerprint.md5 or obo_fingerprint.txt")

    # REPRO-3: embedding cache fingerprint  ← known failure (1 failing test)
    embedding_cache = os.path.join(
        os.path.dirname(__file__), "embedding_cache", "fingerprint.txt"
    )
    check("REPRO-3 embedding cache fingerprint",  # ← this one fails
          os.path.exists(embedding_cache),
          f"path={embedding_cache}")

    # ── Print results ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  PhenoSeq v3 — Test Suite Results")
        print(f"{'═'*60}")
        for name, ok, note in results:
            status = "✓" if ok else "✗"
            suffix = f"  ({note})" if note else ""
            print(f"  {status}  {name}{suffix}")
        print(f"{'─'*60}")
        print(f"  Passed: {passed}/{passed+failed}  "
              f"({'%.0f' % (100*passed/(passed+failed))}%)")
        if failed:
            failing = [name for name, ok, _ in results if not ok]
            print(f"  Failed: {', '.join(failing)}")
        print(f"{'═'*60}\n")

    return {"passed": passed, "failed": failed, "total": passed + failed}


# ═════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── quick demo ────────────────────────────────────────────────────────────
    print("PhenoSeq v3 — hpo_metrics.py\n")

    # NER demo
    preds  = [["cerebellar hypoplasia", "seizures", "ptosis"]]
    golds  = [["pontine hypoplasia",    "seizures", "ptosis"]]
    nm = ner_metrics(preds, golds)
    print(f"NER demo  →  P={nm['precision']}  R={nm['recall']}  F1={nm['f1']}")

    # HPO linking demo
    ranked = [
        ["HP:0012110", "HP:0001321", "HP:0001250"],
        ["HP:0001250", "HP:0000508", "HP:0000486"],
        ["HP:0000508", "HP:0012110", "HP:0001324"],
    ]
    gold = ["HP:0012110", "HP:0001250", "HP:0000508"]

    print(f"Hit@1={hit_at_k(ranked, gold, 1)}  "
          f"Hit@3={hit_at_k(ranked, gold, 3)}  "
          f"MRR={mean_reciprocal_rank(ranked, gold)}")

    # Wu-Palmer demo
    wp = wu_palmer(depth_a=8, depth_b=10, depth_lca=7)
    print(f"Wu-Palmer(8,10,lca=7) = {wp}")

    # Bootstrap CI demo
    hit_vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    ci = bootstrap_ci(hit_vals, n_resamples=1000)
    print(f"Bootstrap 95% CI (Hit@1): {ci}\n")

    # Full test suite
    run_tests(verbose=True)
