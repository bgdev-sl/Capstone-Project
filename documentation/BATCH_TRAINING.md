# Batch-only training and memory guidance (v0.6.0)

## Training mode

This release removes the incremental/online training implementation and its `partial_fit` path. Decision Tree, Random Forest, XGBoost, and MLP are all trained through the shared batch training pipeline.

For the large CICIoT2023 CSV, use `--max-rows N` to select a bounded reservoir sample before training. The loader reads the CSV in small chunks to select a sample from throughout the source file; it does not pass chunks to the model for incremental training. Once selected, the sample is processed and fitted in batch. This preserves a representative sample better than simply taking the first N rows when the file is ordered by collection time or attack class.

**Memory limitation:** batch training on the entire CSV can still exhaust RAM because the selected raw DataFrame and transformed feature matrices are materialized in memory. This release reduces memory requirements by reducing the batch size through `--max-rows`; it does not make full-file batch training fit within arbitrary memory limits.

## Recommended 25,000-row benchmark

```bash
python scripts/benchmark.py \
  --config configs/config.edge.yaml \
  --dataset data/CICIoT2023.csv \
  --max-rows 25000 \
  --sampling-chunk-rows 5000 \
  --reduced-k 24 \
  --output results/benchmark_sample-25_000.csv
```

The sampler's `--sampling-chunk-rows` option controls only how many source CSV rows are read into memory at once while selecting the reservoir sample. It is not a model mini-batch size. Lower values generally reduce sampling memory but increase I/O overhead. The model is still trained with `.fit()` on the selected sample.

The configuration default is `data.sampling_chunk_rows: 10000`. Set this to a value suited to your memory availability or override it on the CLI. The `--max-rows` limit controls the number of rows retained for the actual ML experiment.

## Individual model training

```bash
python scripts/train_models.py \
  --config configs/config.edge.yaml \
  --dataset data/CICIoT2023.csv \
  --model random_forest \
  --version random_forest_v1_sample25000 \
  --max-rows 25000 \
  --sampling-chunk-rows 5000 \
  --feature-k 24
```

## Fair comparison requirements

- Run all four models on the same dataset sample, with the same train/test split seed.
- Keep label normalization, dropped columns, numeric-safety limits, and feature-reduction policy consistent.
- Record the source file hash, sample limit, sampler seed, sampling chunk size, selected label column, class balance, package version, and hardware details.
- Consider materializing the selected sample to a separate CSV, verifying its hash, and reusing it across separate runs if exact cross-run sample identity is required.
- Inspect the train/test class distribution after sampling. Reservoir sampling is uniform over rows, not stratified by class.

## Smoke test

```bash
python scripts/make_dataset.py --rows 2500 --output data/demo_flows.csv
python scripts/benchmark.py \
  --config configs/config.edge.yaml \
  --dataset data/demo_flows.csv \
  --max-rows 2500 \
  --sampling-chunk-rows 500 \
  --reduced-k 8 \
  --output results/demo_batch_benchmark.csv
python -m pytest -q
```

The synthetic demo validates software behavior only. It is not evidence about final CICIoT2023 research outcomes.
