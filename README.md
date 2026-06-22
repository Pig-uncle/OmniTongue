# OmniTongue: A Comprehensive MLLM Benchmark for Automated Tongue Diagnosis in Traditional Chinese Medicine

**Public Release Repository**

## 📖 Introduction

This repository contains the dataset for **OmniTongue**, a comprehensive and fine-grained benchmark designed for Traditional Chinese Medicine (TCM) tongue diagnosis.

Existing tongue diagnosis datasets are often limited by single dimensions and coarse granularity, which can obscure critical pathological information. To bridge this gap, OmniTongue provides expert-validated high-definition tongue images with dense clinical annotations across multiple tongue body and tongue coating dimensions.

The current release contains **1,351 high-definition tongue images** and an expert-validated annotation workbook. A reproducible train/validation/test split is also provided for model development and evaluation.

### Core Features

* **Multidimensional Evaluation**: Dense supervision across 16 clinical dimensions, including tongue body color, size, shape, teeth marks, cracks, coating color, coating thickness, coating distribution, coating texture, and petechiae/ecchymosis.
* **Anatomical Grounding**: Region-level tongue coating supervision across distinct anatomical zones: tip, middle, and root.
* **Fine-Grained Granularity**: Fine-grained diagnostic annotations designed to support precise pathological representation learning.
* **Balanced Benchmark Split**: A released 70/15/15 train/validation/test split balanced across discrete tongue appearance attributes.

## 📊 Dataset Structure

The dataset exhibits a natural long-tailed distribution mirroring authentic clinical scenarios, challenging models to recognize minority-class local lesions rather than exploiting global statistical shortcuts.

Current files are organized as follows:

```text
OminiTongue/
├── label.xlsx
├── origin/
│   └── *.jpg
├── split_70_15_15_seed42/
│   ├── train/
│   │   └── images/
│   ├── val/
│   │   └── images/
│   ├── test/
│   │   └── images/
│   └── labels_split.xlsx
├── RELEASE_NOTES.md
└── README.md
```

### Files

* `origin/`: Contains 1,351 high-resolution tongue images.
* `label.xlsx`: Source annotation workbook with image IDs, tongue appearance labels, report text, syndrome tendency, and diagnostic labels.
* `split_70_15_15_seed42/`: Released benchmark split.
* `split_70_15_15_seed42/labels_split.xlsx`: Annotation workbook with two added metadata columns:
  * `split`: `train`, `val`, or `test`
  * `image_path`: Relative path to the copied image in the released split
* `RELEASE_NOTES.md`: Release summary and validation notes.

## 🔀 Train/Validation/Test Split

The released split uses a fixed 70/15/15 ratio:

| Split | Image count | Percentage |
|---|---:|---:|
| train | 946 | 70.02% |
| val | 203 | 15.03% |
| test | 202 | 14.95% |
| total | 1,351 | 100.00% |

Each image appears in exactly one split. The split is balanced primarily across the 16 discrete tongue appearance attributes rather than the final diagnosis column, because the final diagnosis labels contain many sparse long-tail categories that cannot be strictly balanced across all three splits.

## 🧾 Annotation Dimensions

The released annotation workbook includes the following discrete tongue appearance dimensions:

* `舌色`
* `舌体大小`
* `舌体形态`
* `齿痕程度`
* `裂纹情况`
* `舌苔颜色-全局`
* `舌尖苔色`
* `舌中苔色`
* `舌根苔色`
* `舌苔厚薄-全局`
* `舌尖苔厚薄`
* `舌中苔厚薄`
* `舌根苔厚薄`
* `舌苔分布`
* `舌苔性质`
* `舌面瘀点瘀斑`

Additional text fields provide holistic tongue description, syndrome tendency, and diagnostic annotation.

## ⚙️ Usage

Use `split_70_15_15_seed42/labels_split.xlsx` as the canonical label file for training and evaluation. The `image_path` column points to the corresponding copied image within the split directory.

Example workflow:

1. Read `split_70_15_15_seed42/labels_split.xlsx`.
2. Filter rows by the `split` column.
3. Load images using the relative `image_path` column.
4. Train on `train`, tune on `val`, and report final performance on `test`.

## ✅ Release Validation

The current release was checked for:

* 1,351 source images in `origin/`.
* 1,351 labeled rows in `split_70_15_15_seed42/labels_split.xlsx`.
* Split counts matching `train=946`, `val=203`, and `test=202`.
* No duplicate image assignment across splits.
* No missing copied image paths referenced by `labels_split.xlsx`.
