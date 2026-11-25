# NLP from scratch — AIN442 assignments

Programming assignments for **AIN442 Practicum in Natural Language Processing** (Hacettepe University, Spring 2026). Each one builds a classic NLP component, mostly from scratch, and reports results that can be reproduced from this repository.

| Assignment | Topic | Highlights |
|---|---|---|
| [PA1](PA1/) | Byte-Pair Encoding tokenizer | Pure Python; Turkish character support; learns merge rules and encodes unseen text into token IDs |
| [PA2](PA2/) | N-gram language model | Unigram/bigram/trigram counts, add-1 smoothing, perplexity and text generation on a Turkish corpus; standard library only |
| [PA3](PA3/) | Part-of-speech tagging | HMM tagger with Viterbi decoding vs. a feature-based logistic-regression tagger on the Brown corpus (universal tagset) |
| [PA4](PA4/) | Word embeddings | word2vec neighbours, sentence vectors and semantic search; IMDB sentiment classifier (averaged word2vec + PyTorch MLP) |
| [PA5](PA5/) | Natural language inference: BERT vs. LLMs | Fine-tuned BERT (90.3% on SNLI test) vs. zero-/few-shot qwen2.5:7b and llama3.2:3b via Ollama; invalid-output tracking, latency comparison |

## Setup

PA1 and PA2 need only Python 3. PA3 and PA4 need:

```bash
pip install -r requirements.txt
```

PA4 downloads the pretrained Google News word2vec model (~1.6 GB) through gensim on first use. PA5 is meant to run on a GPU notebook (Kaggle or Colab); see its README.
