# LLM-HM Training

This repository contains the training-only part of LLM-HM, our hallucination mitigation fine-tuning code for vision-language models. It includes the model definitions, DeepSpeed configurations, and training scripts for Qwen3-VL and Ministral. Evaluation code is intentionally excluded from this release.

## Repository Layout

```text
.
|-- ds/
|   |-- ds_z0_config.json
|   |-- ds_z2_config.json
|   |-- ds_z2_offload_config.json
|   |-- ds_z3_config.json
|   `-- ds_z3_offload_config.json
|-- model/
|   |-- ministral_vl_model.py
|   |-- qwen_vl_model.py
|   `-- qwen_vl_moe_model.py
`-- train/
    |-- train_ministral_modified.py
    |-- train_qwen3_vl_moe_modified.py
    `-- train_qwen_modified.py
```

## What This Code Trains

The training scripts add and optimize evidence-based modules for hallucination mitigation. In the recommended setup, the base vision-language model is frozen and only the newly introduced evidence modules are trained.

The repository currently supports:

- `train/train_qwen_modified.py` for dense Qwen3-VL models
- `train/train_qwen3_vl_moe_modified.py` for Qwen3-VL MoE models
- `train/train_ministral_modified.py` for Ministral vision-language models

## Recommended Environment

The original project was developed with the following software stack:

- Python `3.11`
- PyTorch `2.4.1+cu121`
- `datasets==4.4.2`
- `deepspeed==0.15.1`
- `accelerate==1.12.0`
- a recent `transformers` build with Qwen3-VL support

A minimal setup is:

```bash
pip install torch datasets deepspeed accelerate pillow
pip install git+https://github.com/huggingface/transformers
```

## Base Models

Download the base checkpoints separately and point the training scripts to local placeholder paths such as `<QWEN_BASE_MODEL>` or `<MINISTRAL_BASE_MODEL>`.

Recommended upstream model pages:

- Qwen3-VL-2B-Instruct: `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`
- Qwen3-VL-4B-Instruct: `https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct`
- Qwen3-VL-30B-A3B-Instruct: `https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct`
- Ministral-3-3B-Instruct-2512-BF16: `https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16`

## Training Dataset

This repository expects a processed training annotation file and a matching image directory. The code does not require raw upstream datasets at runtime; instead, it reads:

- `<DATA_JSON>`: a JSON file containing training samples
- `<IMAGE_ROOT>`: a directory containing the images referenced by `img`

Each sample should follow this schema:

```json
[
  {
    "id": 1,
    "img": "000000087286.jpg",
    "text": "What is hanging from the traffic light pole?",
    "labels": "A green street sign is hanging from the traffic light pole."
  }
]
```

During preprocessing, the training scripts:

- read the JSON file from `<DATA_JSON>`
- load the image by joining `<IMAGE_ROOT>` and `img`
- resize the image to `224 x 224`
- use `text` as the user-side instruction
- use `labels` as the target response to be generated

### Dataset Used in Our Training

Based on the training artifacts used in the original project, our experiments use a COCO-2017-based visual instruction dataset derived from ShareGPT4V / ShareCaptioner resources and aligned with COCO image files. In practice, we train on a processed subset rather than the raw upstream dumps.

Upstream references:

- ShareGPT4V project: `https://github.com/ShareGPT4Omni/ShareGPT4V`
- ShareGPT4V dataset card: `https://huggingface.co/datasets/Lin-Chen/ShareGPT4V`
- MS COCO download page: `https://cocodataset.org/#download`

If you want to reproduce the same data pipeline, prepare:

- a processed JSON annotation file exported to `<DATA_JSON>`
- the corresponding COCO images under `<IMAGE_ROOT>`

You can also create smaller subsets with the same schema as long as the `img`, `text`, and `labels` fields remain unchanged.

## Training Setup

This release documents only the `Ours` configuration:

| Variant | Description | Key Settings |
| --- | --- | --- |
| `Ours` | All-layer evidence injection with evidence regularization | `--inject_position per_layer --inject_op ours --use_utilization true --lambda_orth 1.0 --lambda_ctr 1.0` |

## Training Commands

All commands below use symbolic paths instead of machine-specific absolute paths:

- `<DATA_JSON>`: processed annotation file
- `<IMAGE_ROOT>`: image directory
- `<MODEL_DIR>`: local base-model directory
- `<OUTPUT_DIR>`: output directory for checkpoints and cache

### Qwen Dense Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 \
  train/train_qwen_modified.py \
  --model_name_or_path <MODEL_DIR> \
  --training_data_path <DATA_JSON> \
  --training_image_dir <IMAGE_ROOT> \
  --output_dir <OUTPUT_DIR> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --bf16 true \
  --logging_steps 2 \
  --remove_unused_columns false \
  --deepspeed ./ds/ds_z2_config.json \
  --finetune_type full \
  --freeze_base_model true \
  --train_evidence_modules true \
  --enable_evidence true \
  --inject_position per_layer \
  --inject_op ours \
  --use_utilization true \
  --evidence_source aligned \
  --gate_layers all \
  --lambda_orth 1.0 \
  --lambda_ctr 1.0
```

### Qwen MoE Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 \
  train/train_qwen3_vl_moe_modified.py \
  --model_name_or_path <MODEL_DIR> \
  --training_data_path <DATA_JSON> \
  --training_image_dir <IMAGE_ROOT> \
  --output_dir <OUTPUT_DIR> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --bf16 true \
  --logging_steps 2 \
  --remove_unused_columns false \
  --deepspeed ./ds/ds_z2_config.json \
  --finetune_type full \
  --freeze_base_model true \
  --train_evidence_modules true \
  --enable_evidence true \
  --inject_position per_layer \
  --inject_op ours \
  --use_utilization true \
  --evidence_source aligned \
  --gate_layers all \
  --lambda_orth 1.0 \
  --lambda_ctr 1.0
```

### Ministral Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 \
  train/train_ministral_modified.py \
  --model_name_or_path <MODEL_DIR> \
  --training_data_path <DATA_JSON> \
  --training_image_dir <IMAGE_ROOT> \
  --output_dir <OUTPUT_DIR> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3 \
  --bf16 true \
  --logging_steps 2 \
  --remove_unused_columns false \
  --deepspeed ./ds/ds_z2_config.json \
  --finetune_type full \
  --freeze_base_model true \
  --train_evidence_modules true \
  --enable_evidence true \
  --inject_position per_layer \
  --inject_op ours \
  --use_utilization true \
  --evidence_source aligned \
  --gate_layers all \
  --lambda_orth 1.0 \
  --lambda_ctr 1.0
```

## Key Arguments

- `--training_data_path`: path placeholder for the processed training JSON file
- `--training_image_dir`: path placeholder for the corresponding image directory
- `--model_name_or_path`: path placeholder for the downloaded base model
- `--output_dir`: directory used for checkpoints and preprocessing cache
- `--inject_position`: set to `per_layer` for all-layer evidence injection
- `--inject_op`: set to `ours` for the evidence fusion operator used in this release
- `--lambda_orth`: weight of the orthogonality regularizer
- `--lambda_ctr`: weight of the contrastive regularizer
- `--aux_layers`: optional comma-separated layer indices for restricting the regularization scope

## Expected Output

The output directory referenced by `<OUTPUT_DIR>` typically contains:

```text
<OUTPUT_DIR>/
|-- checkpoint-*/
|-- preprocessed_cache/
|-- config.json
|-- processor_config.json
`-- tokenizer files
```

The `preprocessed_cache` directory is created automatically after dataset preprocessing and is reused in later runs to reduce startup cost.
