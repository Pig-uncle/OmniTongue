# OmniTongue: A Comprehensive MLLM Benchmark for Automated Tongue Diagnosis in Traditional Chinese Medicine

**Anonymous Repository for MICCAI Double-Blind Review**

## 📖 Introduction
This repository contains the dataset and evaluation codebase for **OmniTongue**, the first comprehensive and fine-grained Visual Question Answering (VQA) benchmark specifically designed for Traditional Chinese Medicine (TCM) tongue diagnosis. 

Existing datasets are often limited by single dimensions and coarse granularity, which obscures critical pathological information. To bridge this gap, OmniTongue integrates 1,351 expert-validated high-definition images and 21,616 multiple-choice instruction pairs. 

### Core Features
* **Multidimensional Evaluation**: Dense supervision across 16 clinical dimensions, including Tongue Body (color, size, shape, teeth marks, cracks, petechiae) and Tongue Coating (color, thickness, distribution, texture).
* **Anatomical Grounding**: Region-level supervision mapping to distinct anatomical zones: Tip (Heart/Lung), Middle (Spleen/Stomach), and Root (Kidney).
* **Fine-Grained Granularity**: 66 distinct diagnostic labels to enforce precise pathological representation learning.

## 📊 Dataset Structure
The dataset exhibits a natural long-tailed distribution mirroring authentic clinical scenarios, challenging models to recognize minority-class local lesions rather than exploiting global statistical shortcuts.

Data files are located in the `data/` directory:
* `images/`: Contains 1,351 anonymized, high-resolution tongue images.
* `annotations/omnitongue_vqa.json`: The structured VQA pairs containing the programmatic mapping of the 16 clinical dimensions into natural language questions and answers.

## ⚙️ Environment Setup
To reproduce the evaluation pipeline, set up the following environment:

```bash
conda create -n omnitongue python=3.10
conda activate omnitongue
pip install -r requirements.txt
