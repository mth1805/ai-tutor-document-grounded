"""Confusion matrices for sufficiency decisions and web fallback routing."""
from __future__ import annotations


def confusion_matrix(rows: list[dict], *, gold_key: str, prediction_key: str) -> dict:
    tp = tn = fp = fn = 0
    for row in rows:
        gold, predicted = bool(row[gold_key]), bool(row[prediction_key])
        if gold and predicted:
            tp += 1
        elif not gold and not predicted:
            tn += 1
        elif not gold and predicted:
            fp += 1
        else:
            fn += 1
    return {
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
        "sufficient_precision": tp / (tp + fp) if tp + fp else None,
        "sufficient_recall": tp / (tp + fn) if tp + fn else None,
        "false_sufficient_rate": fp / (fp + tn) if fp + tn else None,
        "false_insufficient_rate": fn / (fn + tp) if fn + tp else None,
    }


def web_routing_metrics(rows: list[dict]) -> dict:
    correct = sum(bool(row["should_use_web"]) == bool(row["used_web_fallback"]) for row in rows)
    should_web = sum(bool(row["should_use_web"]) for row in rows)
    should_not = len(rows) - should_web
    return {
        "correct_web_triggers": sum(bool(r["should_use_web"]) and bool(r["used_web_fallback"]) for r in rows),
        "false_web_triggers": sum(not bool(r["should_use_web"]) and bool(r["used_web_fallback"]) for r in rows),
        "correct_no_web_decisions": sum(not bool(r["should_use_web"]) and not bool(r["used_web_fallback"]) for r in rows),
        "routing_accuracy": correct / len(rows) if rows else None,
        "web_positive_count": should_web,
        "no_web_count": should_not,
    }
