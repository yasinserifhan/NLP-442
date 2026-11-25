# PA3 — Part-of-speech tagging: HMM vs. logistic regression

Two POS taggers trained and compared on the same split of the Brown corpus, using the 12-tag universal tagset:

1. **HMM tagger with Viterbi decoding** (`hmm_tagger.py`), written from scratch: transition and emission counts, Laplace-smoothed transitions, log-space Viterbi with backpointers, and a fixed probability mass for unknown words.
2. **Feature-based logistic regression** (`logreg_tagger.py`), built with scikit-learn: each token is classified independently from word identity, capitalization, digit flag, 2-character prefix and suffix, and the previous and next words (`DictVectorizer` → `LogisticRegression`).

## Data

Brown corpus from NLTK (`brown.tagged_sents(tagset='universal')`): 57,340 sentences, split 80/20 with `random_state=42`. That gives 45,872 training and 11,468 test sentences, and 231,927 test tokens. The scripts download the corpus automatically.

## Results

| Tagger | Accuracy | Macro F1 | F1 on `X` (rare / foreign words, 329 tokens) |
|---|---:|---:|---:|
| HMM + Viterbi | 0.9572 | 0.92 | 0.56 |
| Logistic regression | **0.9745** | 0.92 | 0.34 |

Full classification reports and sample predictions are in [`results_hmm.txt`](results_hmm.txt) and [`results_logreg.txt`](results_logreg.txt).

**Observations**
- The discriminative model is 1.7 points more accurate overall. The biggest gain is on ADV (F1 0.89 → 0.94), followed by ADJ, ADP, PRT and VERB (each +0.02). In these classes affixes and the neighbouring words are informative, while the HMM sees only the word itself and the previous tag.
- The HMM does better on the rare `X` tag (recall 0.46 vs. 0.21). It models the whole tag sequence, while logistic regression classifies each token on its own and leans toward frequent classes.
- Both taggers are near perfect on closed classes such as punctuation, CONJ and DET.

## Run

```bash
pip install nltk scikit-learn numpy
python hmm_tagger.py      # ~1.5 min on a laptop
python logreg_tagger.py   # ~3 min on a laptop
```
