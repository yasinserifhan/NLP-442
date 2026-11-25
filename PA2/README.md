# PA2 — N-gram language model from scratch

A unigram / bigram / trigram language model in pure Python (standard library only), trained on a small Turkish corpus. It counts n-grams, computes unsmoothed and add-1 (Laplace) smoothed probabilities, scores sentences, measures perplexity and generates text.

## What it does

- **Preprocessing** (as specified in the assignment): regex tokenization, Turkish-aware lowercasing (`I → ı`, `İ → i`), sentence splitting on `. ? !`, and `<s>` / `</s>` padding.
- **Counts and probabilities:** unigram, bigram and trigram counts; unsmoothed and add-1 smoothed probabilities with an extra slot for unknown words (`|V| + 1`).
- **Scoring:** sentence probability (bigram or trigram) in log space, and perplexity on a test file.
- **Generation:** samples the next word from the top-k most frequent bigram followers, weighted by bigram counts.

## Run

```bash
python ngram_lm.py                     # train_corpus.txt -> visible_testCorpus.txt
python ngram_lm.py --train my_corpus.txt --test my_test.txt --seed 0
```

## Results

Output of `python ngram_lm.py` (full output in [`results.txt`](results.txt)):

| | |
|---|---|
| Training corpus | 520 tokens, 279 word types, 40 sentences |

| Model (add-1 smoothing) | Perplexity on `visible_testCorpus.txt` |
|---|---:|
| Unigram | 160.82 |
| Bigram | 157.99 |
| Trigram | 239.36 |

**Why the trigram model is worse here:** the training corpus has only 520 tokens, so almost every trigram context in the test set is unseen. With add-1 smoothing an unseen context gets probability 1 / (|V| + 1) = 1/280 for every word, so trigram perplexity drifts toward 280. Bigram contexts are seen more often, which gives the bigram model a small edge over the unigram model. On a corpus this small, add-1 smoothing moves most of the probability mass to unseen events; a larger corpus or better smoothing (e.g. Kneser–Ney or interpolation) would be needed for the higher-order model to pay off.

Sample generations (top-3 followers, seed 42):

```text
<s> bir dil modelinin test dosyası da temel ilkelerini somutlaştırmayı amaçlar . </s>
<s> bir model davranışını doğrudan etkiler . </s>
```

## Files

| File | Contents |
|---|---|
| `ngram_lm.py` | The `ngramLM` class and a command-line entry point |
| `train_corpus.txt`, `visible_testCorpus.txt` | Turkish training and test corpora provided with the assignment |
| `tinyTestCorpus.txt` | Toy corpus for sanity checks |
| `results.txt` | Output of the command above |

## Changes after submission

The code was cleaned up after grading. On the tiny test corpus, all 30 count and probability values, the sentence list, the vocabulary and the sorted bigrams are identical to the submitted version. The changes:

- The corpus is read line by line, as the assignment specifies; previously the whole file was read at once.
- Bigram and trigram counts are built during training, so the count methods no longer depend on calling `bigrams()` first.
- `sentenceProb` no longer adds `</s>` to the start of a sentence, and the unigram branch now scores tokens rather than indices.
- `perplexity` now reads and tokenizes the test file. Because of this, the perplexity values above differ from the values in the submitted report.
- Generation samples by raw bigram counts, as specified; previously it used smoothed probabilities.
