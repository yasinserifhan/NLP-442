# Custom Byte-Pair Encoding (BPE) Tokenizer

A from-scratch Python implementation of the Byte-Pair Encoding (BPE) algorithm, a foundational subword tokenization technique widely used in modern Large Language Models (LLMs). 

This repository contains a tokenizer that learns merge rules from training text and applies those learned rules to efficiently tokenize unseen test data into subwords and integer sequences (Input IDs).

## 🚀 Features

* **Customizable Vocabulary:** Pre-initialized to support English, Turkish characters (`ç, ğ, ı, ö, ş, ü`), digits, and standard punctuation.
* **Tunable Hyperparameters:** Control the tokenization granularity using `max_merge_count` (how many merge rules to learn) and `topK` (candidate selection strategy).
* **Train and Test Pipeline:**
  * `train_BPE`: Analyzes a training corpus, identifies the most frequent adjacent subword pairs, builds a merge history, and updates the vocabulary.
  * `test_BPE`: Applies the learned merge rules to new text datasets, outputting the final tokenized corpus and its corresponding encoded integer IDs.
* **Cross-Platform Output:** Automatically generates cleanly formatted `.txt` files containing truncated previews of merge lists, vocabularies, and token counts.

## 🛠️ Usage

### Prerequisites
This script uses standard Python libraries (`sys`, `collections.Counter`). No external dependencies (like `pip install ...`) are required. Ensure you have Python 3.x installed.

### Running the Tokenizer
1. Place your text datasets (e.g., `train.txt`, `test.txt`) in the same directory as the script.
2. Run the script from your terminal:

```bash
python template.py
