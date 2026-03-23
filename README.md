# BioMetNet — Genome-to-Metabolism Neural Mapping

A deep learning framework that predicts active metabolic reactions from E. coli genome annotations. Given a set of genes (as a binary presence vector or GFF annotation), BioMetNet predicts which metabolic reactions are catalysed by the organism.

Trains across multiple E. coli strains using universal gene features from [BiGG Models](http://bigg.ucsd.edu/).

## Quickstart

```bash
# Install
uv sync --extra dev

# Option 1: Toy data (fast, for testing)
uv run python scripts/generate_toy_data.py
uv run python scripts/train.py
uv run python scripts/evaluate.py

# Option 2: E. coli iML1515 (single strain, ~1500 genes, ~1100 reactions)
uv run python scripts/prepare_ecoli_data.py
uv run python scripts/train.py --dataset ecoli
uv run python scripts/evaluate.py --dataset ecoli --sweep --pathway-breakdown

# Option 3: E. coli cross-strain (~34 BiGG strain models)
uv run python scripts/prepare_data.py
uv run python scripts/train.py --dataset ecoli_strains
uv run python scripts/evaluate.py --dataset ecoli_strains --sweep
```

## Models

### GenomeClassifier (primary)

Cross-attention transformer for multi-label reaction prediction.

1. **GenomeEncoder** — learned gene embeddings + positional encoding + transformer self-attention. Absent genes are gated to zero.
2. **Reaction queries** — one learnable embedding per reaction.
3. **Cross-attention blocks** — each reaction learns which genes matter by attending to gene encodings.
4. **Classification head** — per-reaction sigmoid output.

Training uses focal BCE loss (gamma=2) with label smoothing, AdamW with warmup-cosine schedule, AMP (mixed precision), and gradient accumulation.

### EcoliStrainClassifier

Same cross-attention architecture but replaces per-gene-ID embeddings with a strain-agnostic **GeneFeatureEncoder** that projects universal gene features (EC number, metabolic subsystem) through a transformer. No positional encoding — genes are treated as a set. Generalises across E. coli strains because features are shared.

### Seq2SeqModel (legacy)

Encoder-decoder transformer that generates reaction ID token sequences autoregressively. Teacher-forced training with cross-entropy loss, greedy decoding at inference. Kept for comparison but the classifier approach is faster and more accurate.

## Prediction from GFF

Predict active reactions for a new genome directly from a GFF3 annotation file:

```bash
uv run python scripts/predict.py \
    --gff path/to/genome.gff.gz \
    --checkpoint checkpoints/best.pt \
    --threshold 0.4
```

## Evaluation

```bash
# Full evaluation with threshold sweep and per-pathway breakdown
uv run python scripts/evaluate.py --dataset ecoli --sweep --pathway-breakdown
```

**Metrics computed:**
- **Reaction-level** — precision, recall, F1
- **Pathway accuracy** — fraction of pathways with exact reaction match
- **Pathway Jaccard** — per-pathway IoU (partial credit)
- **Metabolite coverage** — proxy coverage using reaction sets
- **Per-pathway breakdown** — sorted by F1 (worst pathways first)

The `--sweep` flag tests thresholds from 0.05 to 0.70 to find the optimal operating point.

## Project Structure

```
src/biometnet/
├── data/              # Datasets, vocab, data generation pipelines
│   ├── ecoli_data.py      # E. coli iML1515 pipeline with GPR evaluation
│   ├── strain_data.py     # Cross-strain BiGG pipeline
│   ├── dataset.py         # PyTorch Dataset/collate implementations
│   ├── metabolic_vocab.py # Reaction token vocabulary
│   └── toy_data.py        # Synthetic 40-gene, 10-pathway data
├── model/             # Neural network architectures
│   ├── encoder.py         # GenomeEncoder (gene embeddings + transformer)
│   ├── decoder.py         # MetabolicDecoder (autoregressive)
│   ├── classifier.py      # GenomeClassifier + CrossAttentionBlock
│   ├── feature_encoder.py # GeneFeatureEncoder (strain-agnostic)
│   ├── strain_classifier.py  # EcoliStrainClassifier
│   └── seq2seq.py         # Encoder-decoder wrapper
├── training/          # Training loops and configuration
│   ├── config.py          # ModelConfig, DataConfig, TrainingConfig
│   └── trainer.py         # Trainer, ClassifierTrainer, FocalBCELoss
└── evaluation/
    └── metrics.py         # All evaluation metrics

scripts/               # CLI entry points
├── train.py               # Training (toy/ecoli/ecoli_strains × classifier/seq2seq)
├── evaluate.py            # Evaluation with threshold sweep
├── predict.py             # GFF → reaction predictions
├── prepare_ecoli_data.py  # Download iML1515 + generate training data
├── prepare_data.py        # Download E. coli strain models + build dataset
├── generate_toy_data.py   # Synthetic data for testing
└── download_bigg.py       # Standalone BiGG model downloader

cloud/                 # RunPod deployment scripts
├── setup_pod.sh           # One-time pod setup (uv, deps, GPU check)
├── run_pipeline.sh        # Full train + evaluate pipeline
├── deploy.sh              # rsync project to pod
└── sync_results.sh        # rsync results back from pod

tests/                 # Unit tests (pytest)
```

## Data Pipeline

**E. coli (single strain):** Downloads iML1515 from BiGG → extracts genes, reactions, GPR rules, pathway definitions → generates 30K samples via gene dropout strategies (block, independent, pathway-level) → resamples to 12K balanced samples → 80/10/10 split.

**E. coli (cross-strain):** Downloads ~34 E. coli strain models from BiGG → builds universal reaction set (reactions appearing in ≥2 strains) → extracts strain-agnostic gene features (EC, subsystem) → generates 1000 dropout-augmented samples per strain → saves as padded tensors for fast loading.

**Toy:** 40 genes, 10 synthetic pathways, 1000 samples. Useful for quick iteration and testing.

## Training Configuration

| Setting | Toy | E. coli (single) | E. coli (cross-strain) |
|---------|-----|---------|----------------|
| Model | GenomeClassifier | GenomeClassifier | EcoliStrainClassifier |
| d_model | 128 | 256 | 256 |
| Encoder layers | 2 | 4 | 2 |
| Cross-attn layers | 1 | 1 | 2 |
| ff_dim | 256 | 512 | 512 |
| Batch size | 32 | 8 | 16 (×2 accum = 32) |
| Learning rate | 3e-4 | 3e-4 | 1e-4 |
| Dropout | 0.1 | 0.1 | 0.2 |
| Epochs | 50 | 60 | 30 |
| Early stopping | 7 epochs patience | 7 epochs patience | 7 epochs patience |

Resume training from a checkpoint with `--resume`:

```bash
uv run python scripts/train.py --dataset ecoli --resume
```

## Cloud Deployment (RunPod)

```bash
# Deploy code to pod
bash cloud/deploy.sh

# SSH into pod and run full pipeline
bash cloud/run_pipeline.sh

# Sync results back
bash cloud/sync_results.sh
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2
- COBRApy ≥ 0.29
- NumPy ≥ 1.26

```bash
uv sync --extra dev  # includes pytest
```
