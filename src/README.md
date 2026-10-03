# Memory Agent Implementation

The student scaffold is implemented with a deterministic offline path and
optional live LangChain model construction.

Main modules:

- `config.py`: environment, paths, model settings, and compact thresholds
- `model_provider.py`: provider normalization and lazy model construction
- `memory_store.py`: token estimation, `User.md`, fact extraction, and compaction
- `agent_baseline.py`: thread-only memory
- `agent_advanced.py`: persistent profile plus compact thread memory
- `benchmark.py`: standard and long-context comparisons
- `test_agents.py`: unit, behavior, correction, and dataset recall tests

Run from the repository root:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

No API key is required for tests or benchmarks. Live mode is selected only
when a configured provider has credentials (or when Ollama is selected).
