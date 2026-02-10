# LLM Radiation

Studying how LLMs degrade under simulated radiation — random bit flips in model weights. The hypothesis is that LLMs, due to distributed representations, are more fault-tolerant than deterministic programs and simpler models.

## Motivation

Silent data corruption (SDC) is a growing problem at datacenter scale as feature sizes shrink. While adversarial bit-flip attacks on CNNs and LLMs are well-studied, **nobody has mapped the full degradation curve** for modern LLMs (7B+ parameters) under uniform random bit flips. This project fills that gap.

### Key Research Questions

1. **Where is the critical threshold?** At what bit-flip rate do LLMs catastrophically fail — and is the transition sharp (phase transition) or gradual?
2. **Does scale help?** Are larger models more resilient per-bit than smaller ones?
3. **Dense vs MoE:** Does expert routing in Mixture-of-Experts architectures provide natural fault tolerance?
4. **LLMs vs simpler models:** Are distributed representations genuinely more resilient than logistic regression, CNNs, or BERT-tiny?

## Approach

1. Load open-source models (1B to 13B+, including MoE)
2. Benchmark at baseline (0 flips)
3. Progressively flip random bits in FP16 weights at increasing rates
4. Re-benchmark at each rate across multiple trials
5. Plot degradation curves and fit sigmoids to find critical flip rates
6. Compare across architectures and against control models

## Models

| Model | Parameters | Architecture |
|-------|-----------|-------------|
| TinyLlama-1.1B | 1.1B | Dense |
| Gemma-2B | 2B | Dense |
| Phi-3-mini | 3.8B | Dense |
| Mistral-7B | 7.3B | Dense |
| Gemma-7B | 8.5B | Dense |
| Llama-3.1-8B | 8B | Dense |
| Llama-3.1-13B | 13B | Dense |
| Mixtral-8x7B | 46.7B | MoE |

**Controls:** Logistic regression, small CNN, BERT-tiny

## Benchmarks

MMLU, HellaSwag, ARC-Challenge, TruthfulQA, Winogrande, GSM8K, plus custom perplexity on WikiText-103.

## Project Structure

```
llm-radiation/
├── configs/
│   ├── models/          # Per-model YAML configs (8 models)
│   ├── benchmarks/      # Benchmark suite configs
│   └── sweeps/          # Full experiment sweep configs
├── src/llm_radiation/
│   ├── bitflip/         # Core bit flip engine (FP16 XOR, manifest)
│   ├── models/          # Model registry + control models
│   ├── benchmarks/      # lm-eval bridge, perplexity, control eval
│   ├── experiments/     # Config, runner, adaptive sampling
│   ├── tracking/        # W&B integration
│   ├── infrastructure/  # Lambda Labs API + deployment
│   └── analysis/        # Phase detection + publication plots
├── scripts/             # CLI entry points
├── tests/               # Unit + integration tests
├── notebooks/           # Exploration
└── paper/               # LaTeX materials
```

## Installation

```bash
# Clone and install with uv
git clone https://github.com/traghav/llm-radiation.git
cd llm-radiation
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run a single experiment
uv run python scripts/run_experiment.py configs/sweeps/full_sweep.yaml

# Run a sweep across all models
uv run python scripts/run_sweep.py configs/sweeps/full_sweep.yaml --models-dir configs/models

# Generate paper figures from results
uv run python scripts/generate_plots.py results/ --output-dir paper/figures

# Launch on Lambda Labs GPU
export LAMBDA_API_KEY="your-key"
uv run python scripts/launch_lambda.py configs/sweeps/full_sweep.yaml \
    --ssh-key-name mykey --ssh-key-path ~/.ssh/id_rsa
```

## How the Bit Flip Engine Works

The core engine performs zero-copy FP16 bit manipulation:

```python
from llm_radiation.bitflip.engine import BitFlipEngine
from llm_radiation.models.registry import load_model, snapshot_weights, restore_weights

model, tokenizer = load_model("tinyllama")
snapshot = snapshot_weights(model)  # Save clean weights

engine = BitFlipEngine(seed=42)
manifest = engine.apply_flips(model, flip_rate=1e-5, model_id="tinyllama")
# manifest records every flip for full reproducibility

restore_weights(model, snapshot)  # Restore clean weights
```

Internally: `tensor.view(-1).numpy().view(np.uint16)` reinterprets FP16 memory as uint16, then XOR with `(1 << bit_position)` flips exactly one bit per element. The numpy view shares memory with the torch tensor — zero-copy, in-place.

## Running Tests

```bash
# Unit tests (no GPU needed)
uv run --with pytest pytest tests/test_bitflip/ tests/test_experiments/ -v

# Integration tests (requires GPU + model download)
uv run --with pytest pytest tests/test_integration/ -v -m slow
```

## Key Dependencies

- PyTorch >= 2.2, Transformers >= 4.40
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) >= 0.4.4
- Weights & Biases for tracking
- scipy for sigmoid fitting

## Literature Context

This work addresses gaps in the existing literature:

- **Rakin et al. (ICCV 2019), AttentionBreaker (2024):** Adversarial (targeted) bit flips — we study *random* flips
- **Santos et al. (IEEE 2017-2022):** Real radiation on GPUs running CNNs — we extend to LLMs
- **ISSRE 2023:** First LLM resilience assessment but only BERT/GPT-2/T5 — we scale to 13B+ and MoE
- **Meta (2021), NVIDIA (2024):** SDC is systemic at datacenter scale — we characterize the failure curve

No prior work maps the full flip-rate-to-performance degradation curve for modern LLMs, compares dense vs MoE under random faults, or identifies phase transitions with adaptive sampling.

## License

MIT
