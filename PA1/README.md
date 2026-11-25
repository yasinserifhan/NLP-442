# PA1 — Byte-Pair Encoding tokenizer from scratch

A pure-Python implementation of Byte-Pair Encoding (BPE), the subword tokenization method behind modern LLM tokenizers. It learns merge rules from a training corpus, then applies them to unseen text to produce subword tokens and integer input IDs. No external dependencies.

## How it works

1. **Initial vocabulary:** English and Turkish letters (`ç ğ ı İ ö ş ü`), digits and punctuation.
2. **Pre-tokenization:** the text is split on whitespace and every word is wrapped in boundary markers (`_word_`), so merges can learn word starts and endings.
3. **Training (`train_BPE`):** count adjacent symbol pairs, choose one, add the merged symbol to the vocabulary, rewrite the corpus, repeat `max_merge_count` times.
   - Pairs are ranked by frequency, then by length, then alphabetically.
   - Among the top `topK` pairs, a pair at the start of a word is preferred, then one at the end of a word, otherwise the most frequent.
4. **Encoding (`test_BPE`):** apply the learned merges in order to unseen text, then map each token to its vocabulary ID.

## Example

Configuration 1 (toy corpus, 10 merges, `topK = 1`) learns merges such as `a+a`, `_+w`, `e+_`, `e+r` and `er+e_`, and tokenizes the test sentence *"Once I go there but where is it now?"* as:

```text
['_', 'O', 'n', 'c', 'e_', '_', 'I', '_', '_', 'g', 'o', '_', '_', 't', 'here_', '_', 'b', 'u', 't',
 '_', '_w', 'here_', '_', 'i', 's', '_', '_', 'i', 't', '_', '_', 'n', 'o', 'w', '?_']
```

## Run

```bash
python tokenizer.py
```

This runs the eight configurations in `__main__` (a toy corpus, then two larger corpora with 250 merges and `topK` = 1, 5, 10) and writes `myoutput.txt`, `myoutput1.txt` and `myoutput2.txt`. They match the committed `output*.txt` files exactly. The full run takes under a minute on a laptop.

## Files

| File | Contents |
|---|---|
| `tokenizer.py` | Implementation (`train_BPE`, `test_BPE` and helpers) |
| `train*.txt`, `test*.txt` | Training and test corpora |
| `output*.txt` | Merge lists, vocabularies, tokenized corpora and input IDs for every configuration |
