# Mini CLI (`mini`)

Go CLI to download, manage and chat with Mini locally. No account, no API key, no
remote backend for inference.

- `mini chat` — interactive conversation (Bubble Tea). Works offline once the model is
  installed.
- `mini run "prompt"` — single prompt, stdout = answer, stderr = diagnostics
  (scriptable).
- `mini models list | info | pull` — local registry, sha256 verification, atomic
  install.

Real inference is done by `python/mini_engine.py` (PyTorch + `mini_llm`), spawned as a
child process over JSON lines on stdio. The Go binary never reimplements the network
and never converts weights blindly.

**Full documentation: [Mini-CLI documentation](https://github.com/Mini-llm-lab/Documentation/tree/main/Mini-CLI)**

## Quickstart

Prerequisites:

- Go >= 1.22 (tested with 1.27.1).
- Python 3.11+ with the `Mini-LLM` venv for real inference:
  ```
  cd ../Mini-LLM
  uv sync
  ```
- Model files: native `models/<dir>/` (`*.pt` + `tok.json`) or an HF export
  (`config.json` + `model.safetensors` + `tokenizer.json`), both produced by
  [Mini-LLM](../Mini-LLM).

```
cd Mini-CLI
go build -o mini ./cmd/mini
./mini --help

# mock (no model needed)
./mini run --mock "Hello"

# real local model (the engine and python are auto-detected from ../Mini-LLM)
./mini run --model ../Mini-LLM/models/16k "What is 2+2?" --max-tokens 32
./mini chat --model ../Mini-LLM/models/16k
```

Usage, configuration file, paths, inference details and model storage live in the
[documentation](https://github.com/Mini-llm-lab/Documentation/tree/main/Mini-CLI).
