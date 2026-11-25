# PA4 — Word embeddings: similarity, semantic search and sentiment classification

Built on the pretrained **Google News word2vec** vectors (300-d, loaded through gensim).

**Part I, `word_similarity.py`**
- `replace_with_similar`: replaces chosen words in a sentence with their nearest neighbour, and returns each word's top-5 neighbours.
- `sentence_vector`: builds a sentence embedding by averaging word vectors; out-of-vocabulary words become zero vectors.
- `most_similar_sentences`: ranks the sentences in a file by cosine similarity to a query, a minimal semantic search.
- `analyze_dimension_contributions`: shows which of the 300 dimensions contribute most to the dot product of two words.

**Part II, `sentiment_classifier.py`**
- Encodes each IMDB review as its averaged word2vec vector, ignoring OOV words.
- Trains a small PyTorch MLP on top: 300 → 64 → 1, ReLU, sigmoid, BCE loss, Adam.
- Accuracy, precision, recall and F1 are computed by hand from the confusion counts.

## Results

**Sentiment classification** on a balanced 10,000-review IMDB subset, with a stratified 80/20 split, 5 epochs and seed 42 (see [`results_part2.txt`](results_part2.txt)):

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.832 | 0.829 | 0.836 | 0.833 |

**Semantic search** (from [`results_part1.txt`](results_part1.txt)):

```text
Query: Does Turkey have good universities ?
  0.68122 -> Turkey is among the countries with the most universities .
  0.60277 -> The capital of Turkey ( from now on it is Türkiye ) is Ankara .

Query: What happened to your backpack ?
  0.68521 -> I forgot my backpack on campus and when I got there it was in the same place .
```

**Word substitution:** "NLP is a fascinating field of study and I love learning about it" becomes "NLP is a interesting fields of study and I love teaching about it". The nearest neighbour is not always a drop-in replacement: *learning → teaching* is related but changes the meaning, and *field → fields* changes the grammar.

**Limitation:** averaged word vectors ignore word order and negation ("not good" ≈ "good"), which limits how far this approach can go. A fine-tuned transformer would be the natural next step.

## Changes after submission

- `prepare_classification_data` now accepts both `pos`/`neg` and `positive`/`negative` labels. The included dataset uses the long form, which the submitted mapping turned into missing values.
- Comments were tidied, and `demo_part2.py` fixes the random seeds so the results above are reproducible.

## Run

```bash
pip install gensim numpy pandas torch scikit-learn
python demo_part1.py   # writes results_part1.txt (~1 min after the first download)
python demo_part2.py   # writes results_part2.txt (~1 min)
```

The first run downloads the word2vec model (~1.6 GB) to `~/gensim-data`.

## Files

| File | Contents |
|---|---|
| `word_similarity.py`, `sentiment_classifier.py` | Implementation |
| `demo_part1.py`, `demo_part2.py` | Runnable demos, adapted from the sample scripts provided with the assignment |
| `data/subset10000_IMDB_Dataset.csv` | 10,000-review balanced subset of the IMDB movie review dataset (Maas et al., 2011), provided with the assignment |
| `data/sentences_sample.txt`, `data/word_pairs_sample.txt` | Sample sentences and word pairs |
