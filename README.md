# Efficient Parameter Optimization for Compact Language Models

This repository contains the code and reports for project **"Efficient Parameter Optimization for Compact Language Models"** at POSTECH.

## Repository Structure
```
├── code/
│ ├── 1. Baseline_134.0M.py
│ ├── 2. Double up-projection_148.1M.py
│ ├── 3. SwiGLU_148.1M.py
│ ├── 4. Lanky Architecture_154.0M.py
│ ├── 5. Embedding Sharing_128.2M.py
│ ├── 6. GQA_135.4M.py
│ ├── 7. Layer Sharing_135.4M.py
│ ├── 8. Enhanced Residual_135.4M.py
│ └── 9. Deep Thinking.py
├── Final Paper.pdf
```


## Project Summary

This project explores how to construct a compact and high-performing language model with fewer than 1 billion parameters. The implementation begins with a MobileLLM-style baseline and progressively integrates parameter-efficient techniques to enhance performance while maintaining a small model size.

### Techniques Explored

#### MobileLLM-Inspired Techniques
- **Baseline**: A GPT-2 decoder-based compact LLM architecture.
- **Double Up-Projection**: Enhances FFN expressiveness using dual projection paths.
- **SwiGLU Activation**: Replaces ReLU for better performance in smaller models.
- **Embedding Sharing**: Reuses input/output embeddings to save parameters.
- **GQA (Grouped-Query Attention)**: Reduces key-value heads to optimize attention.
- **Layer Sharing**: Shares parameters across repeated blocks to minimize model size.

#### Additional Techniques Proposed in This Project
- **Enhanced Residual Connection**: Adds learnable α-weighted skip connections to improve gradient flow and dynamic mixing.
- **Deep Thinking**: Introduces iterative multi-pass reasoning across the full input, enabling refined contextual understanding beyond standard forward passes.

## Results

Models were trained on the Wikitext-103 dataset and evaluated using perplexity. Each added technique progressively improved performance, with the **Deep Thinking** model achieving the lowest perplexity.

## Documents

- **Final Paper**: Full write-up of methods, results, and analysis.

## Author

**Youngjun Yu**  
Department of Computer Science and Engineering  
POSTECH  
colin31472@postech.ac.kr
