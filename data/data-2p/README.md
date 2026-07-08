# 2P data

## Download

Download the data from the [eyewire2-data Hugging Face dataset](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main/data-2p).

## Description

Preprocessed calcium traces from OGB-1 recordings, stored as parquet files.

One parquet file per GCL recording field (5 total), plus field-level and outline DataFrames:

| File | Description |
| --- | --- |
| `df_eyewire2_roi_level_GCL{0..4}.parquet` | ROI-level data (traces, quality indices, trigger times, …) |
| `df_eyewire2_field_level.parquet` | Field-level metadata |
| `df_eyewire2_outline.parquet` | Retinal outline data |

Use `eyewire2_functional_analysis.data_loader` to load these files; see the [main README](../../README.md) for examples.
