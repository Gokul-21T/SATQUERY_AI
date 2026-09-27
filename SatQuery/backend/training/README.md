# SatQuery AI — ML Fine-Tuning & Training Pipeline (SIH 26167)

This directory contains the machine learning adaptation and parameter-efficient fine-tuning (PEFT / LoRA) pipeline for **Smart India Hackathon 2026 / ISRO-SAC (PS ID 26167)**.

---

## 1. Why Fine-Tuning Instead of Training from Scratch?

- Foundational Vision-Language Models (VLMs) have billions of parameters and take weeks and huge GPU clusters to train from scratch.
- In accordance with the SIH 26167 guidelines, we adapt existing open remote-sensing VLM backbones (**Qwen2-VL**, **PaliGemma**, or **GeoChat**) using **LoRA (Low-Rank Adaptation)** on the official benchmark datasets:
  - **BigEarthNet.txt**: 464,044 co-registered Sentinel-1 SAR + Sentinel-2 multispectral pairs with 9.6M annotations for optical-radar cross-modal reasoning.
  - **VRSBench**: Grounding referring expressions and predicting bounding boxes.
  - **CDVQA**: Bi-temporal change detection visual question answering.
  - **Cartosat-2S + RISAT**: Target evaluation pairs for ISRO-SAC finale evaluation.

---

## 2. Directory Structure

```text
backend/training/
│
├── dataset_loader.py       # Standardizes BigEarthNet.txt, VRSBench, and CDVQA into instruction pairs
├── train_lora_sih.py       # PyTorch + Hugging Face Transformers + PEFT LoRA training pipeline
├── evaluate_isro_eval.py   # Benchmark evaluation script (mIoU, F1 score, BLEU-4, Accuracy)
└── README.md               # Architecture documentation and presentation defense guide
```

---

## 3. Running the Training Pipeline

### Quick Inspection Mode (Dry-Run / Verification)
Runs without requiring heavy GPU or PyTorch installation, verifying data formatting, parameter counts, and LoRA target configurations:

```bash
cd backend/training
python train_lora_sih.py --dry-run
```

### Full GPU Fine-Tuning
When running on an NVIDIA GPU environment with PyTorch and Transformers installed:

```bash
pip install -r ../requirements-train.txt
python train_lora_sih.py --model_name Qwen/Qwen2-VL-2B-Instruct --epochs 3 --lr 2e-4
```

### Evaluating Against the ISRO-SAC Protocol
To verify the benchmark metrics:

```bash
python evaluate_isro_eval.py
```

---

## 4. Key Parameter Efficiency Metrics
- **Base Model Parameters:** 2.2 Billion
- **Trainable LoRA Parameters:** 14.1 Million (Rank $r=16$, $\alpha=32$)
- **Trainable Weight Ratio:** **0.64%** (156x memory reduction compared to full fine-tuning)
- **Target Projection Layers:** `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`
