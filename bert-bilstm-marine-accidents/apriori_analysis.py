"""Mine causal-factor association rules with Apriori -- the paper's third
component (Section 3.2 "Apriori Algorithm" / Section 4.4 "Apriori Association
Results" of Zhao et al. (2025)).

The paper runs Apriori on its own hand-labeled, multi-label causal-factor
tags per accident report. This project's MAIB dataset only has a single
incident-type label per report, so causal_factors.py first heuristically
re-derives multi-label causal-factor tags per report via keyword matching
against the paper's own 32 causal-factor category names (see that file's
docstring for the important caveat: this is a stand-in for the paper's
manual annotation, not a reproduction of it). This script then mines those
tags exactly the way the paper describes:

    Support(X union Y)    = P(X and Y both present in a report)
    Confidence(X => Y)    = Support(X union Y) / Support(X)
    Lift(X => Y)          = Confidence(X => Y) / Support(Y)

Default thresholds (--min-support 0.008 --min-confidence 0.15) match the
paper's own general-rule thresholds (Section 4.4); pass --min-support 0.01
--min-confidence 0.1 to match the paper's looser causal-chain thresholds
instead.

Usage:
    uv run apriori_analysis.py
    uv run apriori_analysis.py --top-n 15 --output-csv rules.csv
"""
import argparse

import pandas as pd
from datasets import load_dataset
from mlxtend.frequent_patterns import apriori, association_rules

from dataset import DATASET_NAME, clean_text
from causal_factors import extract_causal_factors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-support", type=float, default=0.008)
    p.add_argument("--min-confidence", type=float, default=0.15)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--output-csv", default=None, help="optional path to save all rules as CSV")
    return p.parse_args()


def rule_to_str(rule):
    antecedent = ", ".join(sorted(rule["antecedents"]))
    consequent = ", ".join(sorted(rule["consequents"]))
    return f"{antecedent} -> {consequent}"


def main():
    args = parse_args()

    raw = load_dataset(DATASET_NAME, split="train")
    texts = [clean_text(t) for t in raw["text"]]
    print(f"Loaded {len(texts)} reports from {DATASET_NAME}")

    factor_lists = [extract_causal_factors(t) for t in texts]
    n_tagged = sum(1 for factors in factor_lists if factors)
    avg_factors = sum(len(f) for f in factor_lists) / len(factor_lists)
    print(
        f"Keyword-tagged {n_tagged}/{len(texts)} reports with >=1 causal factor "
        f"(avg {avg_factors:.2f} factors/report)"
    )

    all_categories = sorted({factor for factors in factor_lists for factor in factors})
    print(f"{len(all_categories)} distinct causal-factor categories matched at least once")

    onehot = pd.DataFrame(
        [{category: (category in factors) for category in all_categories} for factors in factor_lists]
    )

    frequent_itemsets = apriori(onehot, min_support=args.min_support, use_colnames=True)
    print(f"\n{len(frequent_itemsets)} frequent itemsets at min_support={args.min_support}")
    if frequent_itemsets.empty:
        print("No frequent itemsets found -- try a lower --min-support.")
        return

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=args.min_confidence
    )
    print(f"{len(rules)} association rules at min_confidence={args.min_confidence}")
    if rules.empty:
        print("No rules found -- try a lower --min-confidence.")
        return

    rules["rule"] = rules.apply(rule_to_str, axis=1)

    print(f"\n=== Top {args.top_n} rules by confidence (paper's Tables 5/6 style) ===")
    by_confidence = rules.sort_values("confidence", ascending=False).head(args.top_n)
    for _, r in by_confidence.iterrows():
        print(f"  {r['rule']:<70} conf={r['confidence']:.2f}  lift={r['lift']:.2f}  support={r['support']:.3f}")

    print(f"\n=== Top {args.top_n} rules by lift (paper's Tables 7/8 style) ===")
    by_lift = rules.sort_values("lift", ascending=False).head(args.top_n)
    for _, r in by_lift.iterrows():
        print(f"  {r['rule']:<70} lift={r['lift']:.2f}  conf={r['confidence']:.2f}  support={r['support']:.3f}")

    if args.output_csv:
        out = rules[["rule", "support", "confidence", "lift", "conviction"]].sort_values(
            "confidence", ascending=False
        )
        out.to_csv(args.output_csv, index=False)
        print(f"\nSaved all {len(rules)} rules to {args.output_csv}")


if __name__ == "__main__":
    main()
