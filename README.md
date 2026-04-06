# Custom Decoder-Only Large Language Model (V2)

A from-scratch, state-of-the-art decoder-only Large Language Model (LLM) implemented in PyTorch. This project demonstrates the fundamental architecture and training pipeline of modern models (equivalent to the GPT-2 Small 124M parameter class), optimized natively for Apple Silicon (MPS) to take full advantage of unified memory architectures.

## Overview

This repository contains the complete pipeline for pre-training a 124-million parameter language model on the HuggingFace FineWeb-Edu dataset. It utilizes OpenAI's `tiktoken` for production-grade tokenization and streams data asynchronously to allow for massive dataset training without exceeding local RAM limits.

## Architecture & Mathematical Theory

The model is built in a highly modular, object-oriented structure implementing modern architectural breakthroughs.

### 1. Root Mean Square Normalization (`RMSNorm`)
Standard Layer Normalization centers data by subtracting the mean and dividing by the variance. RMSNorm is a computational optimization that removes the mean-centering entirely, as scaling by the root mean square is mathematically sufficient and faster to compute.

**Theory & Math:**
For an input vector $x$ of dimension $d$, the RMS normalized output is:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

### 2. SwiGLU Feed Forward Network (`SwiGLUFeedForward`)
This model replaces standard ReLU with a Swish Gated Linear Unit (SwiGLU), which provides a smoother gradient flow and acts as a learned gating mechanism.

**Theory & Math:**
$$\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_3) W_2$$

### 3. Rotary Positional Embeddings (`RotaryPositionalEmbedding`)
Instead of adding absolute positional vectors to token embeddings, RoPE rotates the Query and Key vectors in a complex plane, vastly improving the model's understanding of sequence order and proximity.

**Theory & Math:**
$$f_q(x_m, m) = (x_m \cos m\theta_1 - x_{m+1} \sin m\theta_1, x_m \sin m\theta_1 + x_{m+1} \cos m\theta_1, \dots)$$

### 4. Grouped-Query Attention (`MultiHeadAttention`)
To address memory bandwidth bottlenecks during inference, this module implements Grouped-Query Attention (GQA), where multiple Query heads share a single Key-Value (KV) pair.

**Theory & Math:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 5. The Core Block (`ModernTransformerBlock`)
The forward pass implements two sequential residual connections using a pre-normalization architecture:
$$h = x + \text{Attention}(\text{RMSNorm}(x))$$
$$output = h + \text{SwiGLU}(\text{RMSNorm}(h))$$

---

## Phase 1: Pre-Training the Model

The pre-training pipeline tokenizes the 10-Billion token FineWeb-Edu dataset via a streaming IterableDataset, tracking progress via `tqdm` and implementing dynamic checkpoint resumption.

1. Ensure the required libraries are installed:
   ```bash
   pip install torch datasets tiktoken tqdm
   ```
2. Execute the training script:
   ```bash
   python train.py
   ```
3. **Hardware Note:** The script automatically routes tensor operations to the `mps` device backend for Apple Silicon.

---

## Phase 2: Generation & Chat

You can test the base model's knowledge using the document completion script, or interact with it conversationally using the chat interface (post-instruction tuning).

**Test Base Model Inference:**
```bash
python generate.py
```

**Run Interactive Chat Interface:**
```bash
python chat.py
```