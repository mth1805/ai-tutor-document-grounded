# Answer Evaluation Report

Dataset queries: 24  
Captured runs: 24  
Gemini model: gemini-3.5-flash-lite  
Configured Gemini model: n/a  
Ragas version: 0.4.3

## Deterministic citation evaluation

Citation correctness: 0.780  
Citation completeness: 0.935

## Latency (milliseconds)

| Metric | Count | Mean |
|---|---:|---:|
| ttft_ms | 23 | 2909.609 |
| generation_latency_ms | 23 | 3646.739 |
| failure_latency_ms | 0 | n/a |
| total_answer_latency_ms | 24 | 13989.826 |

## Ragas answer quality

| Metric | Mean |
|---|---:|
| faithfulness | 0.645 |
| answer_relevance | 0.908 |
| context_precision | 0.819 |
| context_recall | 0.978 |

## Limitations

- Web search was disabled; only local fixture evidence was used.
