# Interpretable Routing for OLMoE

This repository contains code for studying an interpretable routing method for Mixture-of-Experts models on `allenai/OLMoE-1B-7B-0125-Instruct`.

The main idea is to insert a small discrete latent state `z` before MoE routing, then analyze whether routing becomes more structured and interpretable under cross-layer constraints.

## Repository Layout

```text
interpretable-routing/
  README.md
  .gitignore
  bbh_base/
    run_bbh_baseline.py
    run_bbh_z_router.py
    analyze_z_usage.py
    compare_bbh_results.py
    setup_env.sh
    download_olmoe_to_scratch.sh
    submit_*.slurm
  z_router/
    z_router.py
    train_z_router.py
    submit_train_z_router.slurm
  data/
    prepare_tulu3_subset.py
    prepare_tulu3_reasoning_30k.py
    submit_prepare_*.slurm
```

## Components

### `bbh_base/`

BBH evaluation and analysis utilities.

- `run_bbh_baseline.py`: run the frozen base OLMoE model on BBH
- `run_bbh_z_router.py`: run a z-router checkpoint on BBH
- `analyze_z_usage.py`: analyze token-level z usage on BBH prompts
- `compare_bbh_results.py`: compare baseline and z-router runs, and generate a summary report

### `z_router/`

z-router implementation and training code.

- `z_router.py`: wraps OLMoE MoE blocks and adds z-conditioned router bias
- `train_z_router.py`: trains only the z-related parameters while freezing the OLMoE backbone

### `data/`

Data preparation scripts.

- `prepare_tulu3_subset.py`: small pilot subset builder
- `prepare_tulu3_reasoning_30k.py`: 30k BBH-oriented reasoning mixture

## Environment

The project uses:

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `pyarrow`
- `pandas`
- `dill`
- `multiprocess`
- `xxhash`
- `fsspec[http]`

A convenience setup script is included:

```bash
cd bbh_base
bash setup_env.sh
```

You can also create your own environment manually as long as the dependencies above are installed.

## Model Download

The included helper script downloads `allenai/OLMoE-1B-7B-0125-Instruct` to a local model directory:

```bash
cd bbh_base
bash download_olmoe_to_scratch.sh
```

If you do not want to use that script, you can place the model anywhere and pass the path through the corresponding command-line arguments.

## Training Data

There are two data-preparation tracks.

### 1. Small pilot subset

For a small pilot dataset:

```bash
python data/prepare_tulu3_subset.py
```

Default output:

```text
data/tulu3_subsets/tulu3_reasoning_small.jsonl
```

### 2. BBH-oriented reasoning 30k

For the larger default reasoning mixture:

```bash
python data/prepare_tulu3_reasoning_30k.py
```

Default output:

```text
data/tulu3_reasoning/tulu3_reasoning_30k.jsonl
data/tulu3_reasoning/tulu3_reasoning_30k.meta.json
```

Default 30k mixture:

- `allenai/tulu-3-sft-personas-math`: 8000
- `allenai/tulu-3-sft-personas-algebra`: 4000
- `allenai/tulu-3-sft-personas-math-grade`: 3000
- `allenai/tulu-3-sft-personas-instruction-following`: 9000
- `allenai/tulu-3-sft-personas-code`: 6000

Override sizes without editing the script:

```bash
python data/prepare_tulu3_reasoning_30k.py \
  --math 16000 \
  --algebra 8000 \
  --math-grade 6000 \
  --instruction-following 18000 \
  --code 12000 \
  --output data/tulu3_reasoning/tulu3_reasoning_60k.jsonl
```

## z-router Training

Train the current default z-router configuration:

```bash
python z_router/train_z_router.py \
  --model-dir /path/to/model \
  --train-jsonl data/tulu3_reasoning/tulu3_reasoning_30k.jsonl \
  --output-dir z_router/checkpoints/<CHECKPOINT_NAME>
```

There is also a Slurm helper script:

```bash
sbatch z_router/submit_train_z_router.slurm
```

Current default configuration:

- layers `6,7,8,9`
- `block_size=4`
- `sharing=cross_layer_shared`
- `sharing_group_size=2`
- `u_sharing=per_layer`
- `num_z=8`
- `tau=1.0`
- `soft_z=true`
- `lambda_balance=0.001`
- `lambda_perturb=0.001`

Default checkpoint name:

```text
z_router_cross_shared_2x2_block6_9_k8_tau1.0_bal0.001_perturb0.001_30k
```

Checkpoint outputs:

```text
z_router/checkpoints/<CHECKPOINT_NAME>/
  z_router_trainable_state.pt
  z_router_config.json
```

## BBH Baseline

Run the baseline:

```bash
python bbh_base/run_bbh_baseline.py \
  --model-dir /path/to/model \
  --output-dir bbh_base/results/bbh_baseline \
  --cache-dir /path/to/bbh_cache
```

Or use the provided Slurm script:

```bash
sbatch bbh_base/submit_bbh_baseline.slurm
```

Baseline outputs:

```text
bbh_base/results/bbh_baseline/
  summary.json
  predictions.jsonl
```

## BBH z-router Evaluation

Run BBH evaluation for a z-router checkpoint:

```bash
python bbh_base/run_bbh_z_router.py \
  --model-dir /path/to/model \
  --checkpoint-dir z_router/checkpoints/<CHECKPOINT_NAME> \
  --output-dir bbh_base/results/<CHECKPOINT_NAME>/bbh_eval \
  --cache-dir /path/to/bbh_cache
```

Run z-usage analysis:

```bash
python bbh_base/analyze_z_usage.py \
  --model-dir /path/to/model \
  --checkpoint-dir z_router/checkpoints/<CHECKPOINT_NAME> \
  --output-dir bbh_base/results/<CHECKPOINT_NAME>/z_usage \
  --cache-dir /path/to/bbh_cache
```

The provided Slurm scripts currently default to an 8-task subset for faster iteration:

- `boolean_expressions`
- `formal_fallacies`
- `logical_deduction_five_objects`
- `tracking_shuffled_objects_seven_objects`
- `multistep_arithmetic_two`
- `reasoning_about_colored_objects`
- `web_of_lies`
- `word_sorting`

To run all BBH tasks when using the Slurm scripts:

```bash
TASKS=all sbatch bbh_base/submit_bbh_z_router.slurm
TASKS=all sbatch bbh_base/submit_analyze_z_usage.slurm
```

## Comparing Baseline and z-router Results

Run compare manually:

```bash
python bbh_base/compare_bbh_results.py \
  --baseline-dir bbh_base/results/bbh_baseline \
  --z-dir bbh_base/results/<CHECKPOINT_NAME>/bbh_eval \
  --output bbh_base/results/<CHECKPOINT_NAME>/compare_vs_baseline.json \
  --z-usage-dir bbh_base/results/<CHECKPOINT_NAME>/z_usage
```

This produces:

```text
bbh_base/results/<CHECKPOINT_NAME>/
  compare_vs_baseline.json
  compare_vs_baseline.md
  compare_vs_baseline.report.json
```

## Results Layout

For each checkpoint, results are grouped under:

```text
bbh_base/results/<CHECKPOINT_NAME>/
  bbh_eval/
  z_usage/
  compare_vs_baseline.json
  compare_vs_baseline.md
  compare_vs_baseline.report.json
```

The baseline reference remains:

```text
bbh_base/results/bbh_baseline/
```

## Notes

- The included `submit_*.slurm` files are convenience wrappers for Slurm-based environments, but the Python entry points can be run directly in any environment.
- Generated JSONL data and experiment outputs are ignored by git.
