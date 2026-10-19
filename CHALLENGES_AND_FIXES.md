# Issues that Became Apparent When CICIoT2023 Dataset was Added

V 0.4.0 demonstrated challenges with memory overflowing due to the size of the dataset. After addressing this, another problem was discovered with handling some numeric data in the CICIoT2023 dataset. When the numeric error was addressed and tested, a TraceBack Error notified of an associated challenge with the array created by the DataFrame. The following documents how these challenges were discovered and how they were addressed.

## 1. Memory Space Limitations

### Discovered Issue

When attempting to load a large dataset into memory for training, the process was stopped due to the file size exceeding available memory on the workstation. As such, the models could not be trained using the full dataset on the current system. 

### The Solution

Model training adds a batch training feature, using the `--max-row N` argument. This can be used to select a bounded reservoir sample before training. The loader reads the CSV in small chunks to select a sample from throughout the source file; it does not pass chunks to the model for incremental training. Once selected, the sample is processed and fitted in batch. This preserves a representative sample better than simply taking the first N rows when the file is ordered by collection time or attack class.

### Method of Implementation

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

To override the memory safe guard, the argument `--allow-large-batch` when calling the `benchmark.py` or `train-models.py` scripts; however it should be noted that this argument could exhaust RAM space.

### Fair Comparison Requirements

- Run all four models on the same dataset sample, with the same train/test split seed.
- Keep label normalization, dropped columns, numeric-safety limits, and feature-reduction policy consistent.
- Record the source file hash, sample limit, sampler seed, sampling chunk size, selected label column, class balance, package version, and hardware details.
- Consider materializing the selected sample to a separate CSV, verifying its hash, and reusing it across separate runs if exact cross-run sample identity is required.
- Inspect the train/test class distribution after sampling. Reservoir sampling is uniform over rows, not stratified by class.

## 2. Numeric Safety

### Discovered Failure

The training pipeline can raise `ValueError: Input X contains infinity or a value too large for dtype('float64')` if the input feature matrix contains `NaN`/`+Inf`/`-Inf`, or if very large finite feature values overflow variance/normalization operations. Flow-derived rates are a plausible source when duration is zero or very small, but the actual offending column must be identified from the dataset and traceback.

### Changes Made

1. The batch `FeaturePipeline` converts numeric inputs to `float64` safely, maps non-finite values to missing values, and allows the imputer to handle the missing data.
2. Raw finite numeric values are clipped to a configurable absolute bound **before** running variance/scaler calculations. The default is `+/-1e12`.
3. Standardized numeric values are clipped to `+/-100` before feature selection, float32 conversion, and ML model input.
4. The configured limits are stored in `configs/config.yaml` and `configs/config.edge.yaml` and the model/feature metadata records the sanitation and clipping counts.
5. The limits are configured under `preprocessing.numeric_safety` in `configs/config.yaml` and `configs/config.edge.yaml`; the batch CLI uses these shared settings.
6. Backward compatibility is retained for previously serialized `FeaturePipeline` model bundles that predate these attributes; absent attributes use the documented defaults.

### Configuration

```yaml
preprocessing:
  numeric_safety:
    max_abs_numeric: 1.0e12
    max_abs_standardized: 100.0
```

This addressed the `ValueError`, but allowed the discovery of another issue in the attempt to fix.

## 3. Read-Only Array

### Cause

The benchmark command was valid. The failure occurred in `FeaturePipeline._sanitize_numeric_input`: `Series.to_numpy()` may return a read-only view when pandas/NumPy storage permits zero-copy access, while the sanitizer subsequently assigns `NaN` into the array. That mutation raises `ValueError: assignment destination is read-only`.

### Fix

The sanitizer now explicitly requests a writable copy with `to_numpy(dtype=np.float64, copy=True)`. It can safely replace `+Inf` and `-Inf`, clip extreme finite values, and leave the original input DataFrame unchanged. A regression test simulates a read-only array and verifies both sanitization and input immutability.

This fix solved the `ValueError` of the read-only array, allowing the system to operate as expected. 
