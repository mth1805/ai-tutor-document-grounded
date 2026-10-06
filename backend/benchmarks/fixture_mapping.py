"""Stable aliases for locally labeled chunks in the retrieval benchmark fixture."""

# The positional indexes point to BENCHMARK_CHUNKS in benchmark_retrieval.py.
FIXTURE_ALIAS_TO_INDEX = {
    "fixture:artificial_intelligence": 0,
    "fixture:machine_learning": 1,
    "fixture:deep_learning": 2,
    "fixture:transformer": 3,
    "fixture:photosynthesis": 4,
    "fixture:chlorophyll": 5,
    "fixture:cellular_respiration": 6,
    "fixture:newton_laws": 7,
    "fixture:relational_database": 8,
    "fixture:hnsw": 9,
}
INDEX_TO_FIXTURE_ALIAS = {index: alias for alias, index in FIXTURE_ALIAS_TO_INDEX.items()}


def fixture_alias(chunk_index: int) -> str:
    return INDEX_TO_FIXTURE_ALIAS.get(chunk_index, f"fixture:chunk_{chunk_index}")
