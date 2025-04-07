# Cross-Lingual Word Embedding Alignment (English-Hindi)

**Author**: Abhiraj Rananajay Singh  
**Date**: *Insert Date*  

This repository contains code and documentation for aligning English and Hindi word embeddings into a shared cross-lingual space. The alignment is performed using a supervised orthogonal Procrustes method (and optionally an unsupervised adversarial approach) on FastText embeddings, evaluated via translation accuracy on a bilingual dictionary.

---

## Table of Contents
1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Supervised Alignment (Procrustes)](#supervised-alignment-procrustes)
4. [Unsupervised Alignment (Optional)](#unsupervised-alignment-optional)
5. [Evaluation](#evaluation)
6. [Ablation Study](#ablation-study)
7. [Results](#results)
8. [References](#references)

---

## Overview

Cross-lingual word embeddings allow words from different languages to be represented in a common vector space. Here, we align:
- **English** FastText embeddings (top 100k words)
- **Hindi** FastText embeddings (top 100k words)

We use:
- A bilingual dictionary (from the [MUSE](https://github.com/facebookresearch/MUSE) dataset) to supervise training.
- The orthogonal Procrustes solution to learn a rotation matrix mapping English embeddings into the Hindi space.
- (Optional) An adversarial + refinement approach (MUSE-like) that does not initially require parallel data.

Key tasks include:
- Loading and normalizing monolingual embeddings
- Learning a linear mapping
- Evaluating via translation accuracy (Precision@1, Precision@5)

---

## Data Preparation

1. **Word Embeddings**:  
   - We use [FastText](https://fasttext.cc/) 300-dimensional vectors for English (`cc.en.300.vec`) and Hindi (`cc.hi.300.vec`).  
   - Limit the vocabulary to the top 100,000 words per language for feasibility.

2. **Bilingual Lexicon**:  
   - From the [MUSE English-Hindi dictionary](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries), we gather a list of word pairs `(en_word, hi_word)`.  
   - Split these pairs into train (e.g., 17k) and test (e.g., 1.5k).

3. **Normalization**:  
   - Each word vector is normalized to unit length to ensure that cosine similarity can be computed as a dot product.

Sample code snippet for loading and normalizing:

```python
import numpy as np

def load_embeddings(file_path, top_n=100000):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()  # may skip header if file has one
        for i, line in enumerate(f):
            if i >= top_n:
                break
            parts = line.rstrip().split(' ')
            if len(parts) < 2:
                continue
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            embeddings[word] = vec
    return embeddings

# Load top 100k English & Hindi word embeddings
eng_embeddings = load_embeddings("cc.en.300.vec", top_n=100000)
hi_embeddings  = load_embeddings("cc.hi.300.vec", top_n=100000)

# Normalize
for emb in [eng_embeddings, hi_embeddings]:
    for word, vec in emb.items():
        emb[word] = vec / (np.linalg.norm(vec) + 1e-9)
