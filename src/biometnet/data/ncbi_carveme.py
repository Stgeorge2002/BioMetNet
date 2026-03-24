"""NCBI genome download and CarveMe model reconstruction pipeline.

Downloads E. coli protein FASTAs from NCBI and reconstructs genome-scale
metabolic models using CarveMe. Output SBML models are directly loadable
by COBRApy for integration with the existing strain_data pipeline.

Usage:
    from biometnet.data.ncbi_carveme import download_ncbi_genomes, run_carveme_batch

    fasta_paths = download_ncbi_genomes(out_dir="data/raw/ncbi", max_genomes=100)
    model_paths = run_carveme_batch(fasta_paths, out_dir="data/raw/carveme_models")
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# NCBI Datasets API
# ---------------------------------------------------------------------------

_NCBI_DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2"
_TAXON_ECOLI = "562"  # NCBI Taxonomy ID for Escherichia coli


def fetch_ecoli_genome_catalog(
    assembly_level: str = "complete_genome",
    max_genomes: int | None = None,
) -> list[dict]:
    """Fetch catalog of E. coli genomes from NCBI Datasets API.

    Returns a list of genome records with accession, organism name,
    assembly info, and annotation metadata.
    """
    url = f"{_NCBI_DATASETS_API}/genome/taxon/{_TAXON_ECOLI}/dataset_report"
    params: dict[str, Any] = {
        "filters.assembly_level": assembly_level,
        "filters.assembly_source": "RefSeq",
        "page_size": min(max_genomes or 1000, 1000),
    }

    all_records: list[dict] = []
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        reports = data.get("reports", [])
        for report in reports:
            accession = report.get("accession", "")
            assembly_info = report.get("assembly_info", {})
            organism = report.get("organism", {})

            record = {
                "accession": accession,
                "organism_name": organism.get("organism_name", ""),
                "infraspecific_name": organism.get("infraspecific_names", {}).get("strain", ""),
                "assembly_level": assembly_info.get("assembly_level", ""),
                "assembly_name": assembly_info.get("assembly_name", ""),
                "submission_date": assembly_info.get("submission_date", ""),
                "refseq_category": assembly_info.get("refseq_category", ""),
            }
            all_records.append(record)

        if max_genomes and len(all_records) >= max_genomes:
            all_records = all_records[:max_genomes]
            break

        page_token = data.get("next_page_token")
        if not page_token:
            break

        time.sleep(0.3)  # polite rate-limiting

    return all_records


def _deduplicate_strains(records: list[dict]) -> list[dict]:
    """Keep one genome per unique strain name.

    Many NCBI entries are re-sequenced versions of the same strain.
    We keep the most recent submission for each unique strain.
    """
    strain_best: dict[str, dict] = {}
    for rec in records:
        strain = rec["infraspecific_name"] or rec["organism_name"]
        existing = strain_best.get(strain)
        if existing is None:
            strain_best[strain] = rec
        else:
            # Keep more recent submission
            if rec["submission_date"] > existing["submission_date"]:
                strain_best[strain] = rec
    return list(strain_best.values())


def download_protein_fasta(
    accession: str,
    out_dir: Path,
    timeout: int = 300,
) -> Path | None:
    """Download protein FASTA for one genome from NCBI Datasets API.

    Returns path to the decompressed .faa file, or None on failure.
    """
    fasta_path = out_dir / f"{accession}.faa"
    if fasta_path.exists() and fasta_path.stat().st_size > 0:
        return fasta_path

    gz_path = out_dir / f"{accession}.faa.gz"
    if gz_path.exists() and gz_path.stat().st_size > 0:
        # Decompress cached gz
        with gzip.open(gz_path, "rb") as f_in, open(fasta_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return fasta_path

    # Download via NCBI Datasets API — get the protein FASTA
    # The datasets v2 download endpoint gives a zip; instead use direct FTP-style
    url = (
        f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        f"{accession}/download"
    )
    params = {
        "include_annotation_type": "PROT_FASTA",
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout, stream=True)
        resp.raise_for_status()

        # The response is a zip file containing the protein FASTA
        zip_path = out_dir / f"{accession}_download.zip"
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract protein FASTA from the zip
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            # Find the protein FASTA inside
            protein_files = [
                n for n in zf.namelist()
                if n.endswith("protein.faa") or n.endswith("protein.faa.gz")
            ]
            if not protein_files:
                zip_path.unlink(missing_ok=True)
                return None

            target = protein_files[0]
            data = zf.read(target)

            if target.endswith(".gz"):
                import io
                with gzip.open(io.BytesIO(data), "rb") as gf:
                    data = gf.read()

            fasta_path.write_bytes(data)

        zip_path.unlink(missing_ok=True)
        return fasta_path

    except Exception:
        # Clean up partial downloads
        gz_path.unlink(missing_ok=True)
        fasta_path.unlink(missing_ok=True)
        return None


def download_ncbi_genomes(
    out_dir: str | Path = "data/raw/ncbi",
    max_genomes: int | None = None,
    deduplicate: bool = True,
    assembly_level: str = "complete_genome",
) -> list[Path]:
    """Download E. coli protein FASTAs from NCBI.

    1. Fetches genome catalog from NCBI Datasets API
    2. Deduplicates by strain name (keeps most recent)
    3. Downloads protein FASTA for each genome

    Returns list of paths to .faa files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save/load catalog for resume capability
    catalog_path = out_dir / "genome_catalog.json"

    if catalog_path.exists():
        print("  Loading cached genome catalog...")
        records = json.loads(catalog_path.read_text())
    else:
        print(f"  Fetching E. coli genome catalog from NCBI "
              f"(assembly_level={assembly_level})...")
        records = fetch_ecoli_genome_catalog(
            assembly_level=assembly_level,
            max_genomes=max_genomes * 3 if max_genomes else None,
        )
        catalog_path.write_text(json.dumps(records, indent=2))
        print(f"  Found {len(records)} genomes in NCBI")

    if deduplicate:
        before = len(records)
        records = _deduplicate_strains(records)
        print(f"  Deduplicated: {before} -> {len(records)} unique strains")

    if max_genomes and len(records) > max_genomes:
        records = records[:max_genomes]

    print(f"  Downloading protein FASTAs for {len(records)} genomes...")
    fasta_dir = out_dir / "fastas"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    failed = 0
    for i, rec in enumerate(records):
        acc = rec["accession"]
        strain = rec["infraspecific_name"] or "unknown"
        print(f"  [{i+1}/{len(records)}] {acc} ({strain})...",
              end=" ", flush=True)

        path = download_protein_fasta(acc, fasta_dir)
        if path is not None:
            size_kb = path.stat().st_size // 1024
            print(f"OK ({size_kb}KB)")
            paths.append(path)
        else:
            print("FAILED")
            failed += 1

        # Rate limit: NCBI allows 3 requests/sec without API key, 10 with
        time.sleep(0.5)

    print(f"\n  Downloaded {len(paths)} FASTAs ({failed} failed)")
    return paths


# ---------------------------------------------------------------------------
# CarveMe model reconstruction
# ---------------------------------------------------------------------------


def _check_carveme_available() -> bool:
    """Check that CarveMe is installed and accessible."""
    try:
        result = subprocess.run(
            ["carve", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_solver() -> str | None:
    """Auto-detect the best available LP solver for CarveMe."""
    for solver in ("cplex", "gurobi", "scip", "glpk"):
        try:
            if solver == "scip":
                import pyscipopt  # noqa: F401
                return solver
            elif solver == "glpk":
                import swiglpk  # noqa: F401
                return solver
            elif solver == "cplex":
                import cplex  # noqa: F401
                return solver
            elif solver == "gurobi":
                import gurobipy  # noqa: F401
                return solver
        except ImportError:
            continue
    return None


def run_carveme_single(
    fasta_path: Path,
    output_path: Path,
    universe: str = "gramneg",
    solver: str | None = None,
    timeout_seconds: int = 1200,
) -> tuple[bool, str]:
    """Run CarveMe on a single protein FASTA file.

    Args:
        fasta_path: Path to input .faa file
        output_path: Path for output .xml SBML model
        universe: CarveMe universe template (gramneg for E. coli)
        solver: LP solver (None = let CarveMe decide)
        timeout_seconds: Max time per model (default 20 min)

    Returns (success, error_message) tuple.
    """
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True, ""  # already built

    cmd = [
        "carve",
        str(fasta_path),
        "--output", str(output_path),
        "--universe", universe,
    ]
    if solver:
        cmd += ["--solver", solver]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        combined = (
            (result.stderr or "") + "\n" + (result.stdout or "")
        ).strip()
        if result.returncode != 0:
            output_path.unlink(missing_ok=True)
            return False, combined or f"exit code {result.returncode}"
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True, ""
        output_path.unlink(missing_ok=True)
        return False, combined or "Output file missing or too small"
    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        return False, f"Timed out after {timeout_seconds}s"
    except FileNotFoundError:
        raise RuntimeError(
            "CarveMe not found. Install with: uv pip install carveme"
        )


def run_carveme_batch(
    fasta_paths: list[Path],
    out_dir: str | Path = "data/raw/carveme_models",
    universe: str = "gramneg",
    timeout_per_model: int = 1200,
) -> list[Path]:
    """Run CarveMe on a batch of protein FASTA files.

    Resumes from where it left off — already-built models are skipped.
    Returns paths to successfully built SBML models.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _check_carveme_available():
        raise RuntimeError(
            "CarveMe not found. Install with:\n"
            "  uv pip install carveme\n"
            "Or add to your project: uv add --extra carveme"
        )

    solver = _detect_solver()
    print(f"  Available solver: {solver or 'none detected — using CarveMe default'}")

    model_paths: list[Path] = []
    failed = 0
    skipped = 0
    first_error: str | None = None

    for i, fasta in enumerate(fasta_paths):
        model_name = fasta.stem  # e.g. GCF_000005845.2
        model_path = out_dir / f"{model_name}.xml"

        if model_path.exists() and model_path.stat().st_size > 1000:
            print(f"  [{i+1}/{len(fasta_paths)}] {model_name}... "
                  f"CACHED", flush=True)
            model_paths.append(model_path)
            skipped += 1
            continue

        print(f"  [{i+1}/{len(fasta_paths)}] {model_name}... ",
              end="", flush=True)

        # Don't pass explicit solver — let CarveMe use its own detection
        success, err_msg = run_carveme_single(
            fasta, model_path, universe=universe,
            solver=None,
            timeout_seconds=timeout_per_model,
        )

        if success:
            size_kb = model_path.stat().st_size // 1024
            print(f"OK ({size_kb}KB)")
            model_paths.append(model_path)
        else:
            print("FAILED")
            failed += 1
            if failed <= 3 and err_msg:
                # Print first 3 errors for diagnosis
                for line in err_msg.splitlines()[:5]:
                    print(f"    | {line}")
            if failed == 3:
                print("    (suppressing further error details)")

    print(f"\n  Built {len(model_paths)} models "
          f"({skipped} cached, {failed} failed)")
    return model_paths


# ---------------------------------------------------------------------------
# Full NCBI + CarveMe pipeline
# ---------------------------------------------------------------------------


def build_ncbi_carveme_models(
    ncbi_dir: str | Path = "data/raw/ncbi",
    carveme_dir: str | Path = "data/raw/carveme_models",
    max_genomes: int | None = None,
    deduplicate: bool = True,
) -> list[Path]:
    """End-to-end: download NCBI genomes and build CarveMe models.

    This is the main entry point. Returns paths to SBML models ready
    for `prepare_strain_dataset()`.
    """
    print("=" * 60)
    print("  NCBI + CarveMe Model Reconstruction Pipeline")
    print("=" * 60)

    print("\n--- Phase 1: Downloading protein FASTAs from NCBI ---")
    fasta_paths = download_ncbi_genomes(
        out_dir=ncbi_dir,
        max_genomes=max_genomes,
        deduplicate=deduplicate,
    )

    if not fasta_paths:
        print("No FASTAs downloaded. Aborting.")
        return []

    print(f"\n--- Phase 2: Building metabolic models with CarveMe ---")
    print(f"  This may take a while (~5-15 min per model)...")
    model_paths = run_carveme_batch(
        fasta_paths,
        out_dir=carveme_dir,
    )

    print(f"\n--- Pipeline complete: {len(model_paths)} models ready ---")
    return model_paths
