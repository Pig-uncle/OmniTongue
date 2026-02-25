# 👅 OmniTongue: A Multimodal Large Language Model for Comprehensive Tongue Diagnosis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![MICCAI 2026](https://img.shields.io/badge/Paper-MICCAI%202026-orange.svg)](#)

**OmniTongue** 是一个专为舌象诊断设计的多模态大模型框架。它结合了先进的视觉-语言模型 (VLM) 与深度中医 (TCM) 医学知识，旨在通过多维度特征提取和检索增强生成 (RAG) 技术，实现精准、可解释的自动舌诊分析。

---

## ✨ 核心特性

* **🏆 权威数据集支持：** 基于 **Tongue-1K** 数据集构建，包含 1,351 张经过专家严格标注的高质量舌象图像。
* **👁️ 多模态理解：** 不仅能识别颜色、苔质、齿痕等视觉特征，还能结合患者主诉进行综合推理。
* **🔍 检索增强生成 (RAG)：** 集成医学知识库，通过 RAG 技术显著降低模型幻觉，提供有据可依的诊断建议。
* **📊 综合基准测试：** 针对舌象分类、分割及描述生成任务建立了完整的 Benchmark 体系。
* **🛠️ 易于扩展：** 模块化设计，支持快速适配不同的多模态底座（如 LLaVA, Qwen-VL 等）。

---

## 📅 更新日志

- **[2026-06]** 发布 OmniTongue 源代码与预训练模型权重。
- **[2026-02]** 提交至 **MICCAI 2026**。
- **[2026-01]** 完成 **Tongue-1K** 数据集标注与整理。

---

## 🚀 快速开始

### 1. 环境准备
```bash
git clone [https://github.com/YourUsername/OmniTongue.git](https://github.com/YourUsername/OmniTongue.git)
cd OmniTongue
pip install -r requirements.txt
