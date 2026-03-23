# BioMetNet — Genome-to-Metabolism Neural Mapping

A machine learning framework that maps bacterial genome assemblies and gene annotations to metabolic systems, represented as canonical reaction token sequences.

## Quickstart

```bash
# Install dependencies
uv sync --extra dev

# Generate synthetic training data
uv run python scripts/generate_toy_data.py

# Train on toy data
uv run python scripts/train.py

# Evaluate
uv run python scripts/evaluate.py
```

## Architecture

**Encoder-decoder transformer** that learns genome → metabolism mappings:

- **Input**: Binary gene-presence vector (length = number of genes)
- **Output**: Ordered sequence of reaction ID tokens representing the organism's metabolic program
- **Training**: Teacher-forced cross-entropy on reaction token sequences
- **Inference**: Greedy or beam-search decoding

## Project Structure

```
src/biometnet/
├── data/          # Data loading, vocab, toy data generation
├── model/         # Encoder, decoder, seq2seq architecture
├── training/      # Config, training loop
└── evaluation/    # Metrics (reaction/pathway/metabolite level)
scripts/           # CLI entry points
tests/             # Unit tests
```

## Metabolic Syntax

Metabolism is encoded as a sorted sequence of reaction ID tokens:

```
<BOS> ACONTa ACONTb AKGDH CS FUM ICDHyr MDH SUCOAS <EOS>
```

This canonical ordering allows the problem to be framed as sequence generation.
