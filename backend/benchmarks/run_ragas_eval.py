"""Optional LLM-as-a-judge scoring for prepared answers; never runs in pytest."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path


def prepare_samples(records: list[dict]) -> list[dict]:
    required = {"question", "answer", "contexts", "reference"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing: {', '.join(sorted(missing))}")
        if not isinstance(record["contexts"], list):
            raise ValueError(f"record {index}: contexts must be a list of strings")
        if not all(isinstance(context, str) for context in record["contexts"]):
            raise ValueError(f"record {index}: contexts must contain strings")
    return records


async def score_records(records: list[dict], model: str) -> dict:
    """Use the current Ragas v0.4 collections API; all metrics call the evaluator LLM."""
    try:
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -r benchmarks/requirements-ragas.txt") from exc
    from openai import AsyncOpenAI

    llm = llm_factory(model, client=AsyncOpenAI())
    metrics = {"faithfulness": Faithfulness(llm=llm),
               "answer_relevance": AnswerRelevancy(llm=llm),
               "context_precision": ContextPrecision(llm=llm),
               "context_recall": ContextRecall(llm=llm)}
    scores = {name: [] for name in metrics}
    for record in records:
        args = {"user_input": record["question"], "response": record["answer"],
                "retrieved_contexts": record["contexts"], "reference": record["reference"]}
        for name, metric in metrics.items():
            result = await metric.ascore(**args)
            scores[name].append(float(result.value))
    return {name: {"mean": sum(values) / len(values) if values else None,
                   "scores": values} for name, values in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSONL of saved question/answer/context/reference records")
    parser.add_argument("--model", default=os.environ.get("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest_ragas_results.json"))
    args = parser.parse_args()
    records = prepare_samples([json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()])
    if not records:
        raise SystemExit("No answer records found.")
    report = asyncio.run(score_records(records, args.model))
    output = {"sample_count": len(records), "evaluator_model": args.model,
              "external_llm_evaluator_used": True, "metrics": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
