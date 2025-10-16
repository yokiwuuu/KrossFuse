# RP-KrossFuse

Official Code for **"Fusing Cross-modal and Uni-modal Representations: A Kronecker Product Approach"**

---

## 📖 Overview

This repository contains the official implementation of our paper on efficient feature fusion using Kronecker products and random projection. We propose a novel method to combine representations from cross-modal (CLIP) and uni-modal (DINOv2) vision transformers for improved image classification performance while maintain the cross modal alignment performance.

**Key Features:**
- 🔥 Efficient Kronecker product-based feature fusion
- ⚡ Random projection for scalable dimensionality reduction
- 🎯 Strong performance on multiple benchmark datasets
- 💾 Feature caching for fast experimentation

---

## 🚀 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yokiwuuu/krossfuse.git
cd krossfuse

# Create virtual environment
conda create -n krossfuse python=3.9
conda activate krossfuse

# Install dependencies
pip install -r requirements.txt
```

---

### Command Line Arguments

```bash
python main.py --dataset <dataset_name> \
               --clip_model <clip_variant> \
               --dino_model <dino_variant> \
               --n_components <projection_dim> \
               --batch_size <batch_size>
```

---

### Acknowledgments

This work builds upon:
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- [OpenCLIP](https://github.com/mlfoundations/open_clip) by LAION
