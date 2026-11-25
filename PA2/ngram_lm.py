# -*- coding: utf-8 -*-
"""
N-gram language model (unigram / bigram / trigram) from scratch.

AIN442 Programming Assignment 2. Cleaned-up version of the submitted code;
see README.md for the list of post-submission fixes.
"""

import argparse
import math
import random
import re
from collections import Counter, defaultdict

# Tokenization regex given in the assignment.
TOKEN_PATTERN = r"""(?x)
(?:[A-ZÇĞIİÖŞÜ]\.)+
| \d+(?:\.\d*)?(?:\’\w+)?
| \w+(?:-\w+)*(?:\’\w+)?
| \.\.\.
| [][,;.?():_!#^+$%&><|/{()=}\"\’\\\"\‘-]
"""

SENTENCE_END = {".", "!", "?"}
BOS, EOS = "<s>", "</s>"

# Turkish alphabetical order for sorting n-grams with equal counts.
_TR_SOURCE = "abcçdefgğhıijklmnoöprsştuüvyz"
_TR_TARGET = "abcdefghijklmnopqrstuvwxyzabc"
_TR_TABLE = str.maketrans(_TR_SOURCE, _TR_TARGET)


def tr_lower(token):
    """Lowercase with Turkish dotted/dotless I handled first (I -> ı, İ -> i)."""
    return token.replace("I", "ı").replace("İ", "i").lower()


def tokenize_file(fn):
    """Read a UTF-8 file line by line and return padded sentences (lists of tokens)."""
    sentences = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            tokens = [tr_lower(t) for t in re.findall(TOKEN_PATTERN, line)]
            if not tokens:
                continue  # skip lines that produce no tokens
            current = []
            for tok in tokens:
                current.append(tok)
                if tok in SENTENCE_END:
                    sentences.append([BOS] + current + [EOS])
                    current = []
            if current:  # the last sentence of a line may lack a final punctuation mark
                sentences.append([BOS] + current + [EOS])
    return sentences


class ngramLM:
    """N-gram language model with unsmoothed and add-1 smoothed probabilities."""

    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        self.ngrams = Counter()          # unigram counts
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self._followers = {}             # word -> [(next_word, count)] sorted by count

    # ------------------------------------------------------------------ training
    def trainFromFile(self, fn):
        self.sentences = tokenize_file(fn)
        for sent in self.sentences:
            self.ngrams.update(sent)
            self.bigram_counts.update(zip(sent, sent[1:]))
            self.trigram_counts.update(zip(sent, sent[1:], sent[2:]))
        self.numOfTokens = sum(self.ngrams.values())
        self.sizeOfVocab = len(self.ngrams)
        self.numOfSentences = len(self.sentences)

        followers = defaultdict(list)
        for (w1, w2), c in self.bigram_counts.items():
            followers[w1].append((w2, c))
        self._followers = {
            w: sorted(lst, key=lambda x: (-x[1], x[0].translate(_TR_TABLE)))
            for w, lst in followers.items()
        }

    # ------------------------------------------------------- vocabulary / n-grams
    def vocab(self):
        return sorted(self.ngrams.items(), key=lambda x: (-x[1], x[0].translate(_TR_TABLE)))

    def _sorted_ngrams(self, counts):
        return sorted(counts.items(),
                      key=lambda x: (-x[1], [w.translate(_TR_TABLE) for w in x[0]]))

    def bigrams(self):
        return self._sorted_ngrams(self.bigram_counts)

    def trigrams(self):
        return self._sorted_ngrams(self.trigram_counts)

    # ------------------------------------------------------------------ counts
    def unigramCount(self, word):
        return self.ngrams[word]

    def bigramCount(self, bigram):
        return self.bigram_counts[tuple(bigram)]

    def trigramCount(self, trigram):
        return self.trigram_counts[tuple(trigram)]

    # --------------------------------------------------- unsmoothed probabilities
    def unigramProb(self, word):
        return self.unigramCount(word) / self.numOfTokens if self.numOfTokens else 0

    def bigramProb(self, bigram):
        w1 = bigram[0]
        if self.unigramCount(w1) == 0:
            return 0
        return self.bigramCount(bigram) / self.unigramCount(w1)

    def trigramProb(self, trigram):
        context = (trigram[0], trigram[1])
        if self.bigramCount(context) == 0:
            return 0
        return self.trigramCount(trigram) / self.bigramCount(context)

    # ------------------------------------------------ add-1 smoothed probabilities
    def unigramProb_SmoothingUNK(self, word):
        return (self.unigramCount(word) + 1) / (self.numOfTokens + self.sizeOfVocab + 1)

    def bigramProb_SmoothingUNK(self, bigram):
        # An unknown w1 gives 1 / (|V| + 1), as the assignment requires.
        return (self.bigramCount(bigram) + 1) / (self.unigramCount(bigram[0]) + self.sizeOfVocab + 1)

    def trigramProb_SmoothingUNK(self, trigram):
        # An unknown context (w1, w2) gives 1 / (|V| + 1).
        context = (trigram[0], trigram[1])
        return (self.trigramCount(trigram) + 1) / (self.bigramCount(context) + self.sizeOfVocab + 1)

    # ---------------------------------------------------------- sentence scoring
    def _log_terms(self, sent, model):
        """Log-probabilities of the scored tokens of one padded sentence."""
        if model == "unigram":
            return [math.log(self.unigramProb_SmoothingUNK(w)) for w in sent[1:]]
        if model == "bigram":
            return [math.log(self.bigramProb_SmoothingUNK((sent[i - 1], sent[i])))
                    for i in range(1, len(sent))]
        if model == "trigram":
            return [math.log(self.trigramProb_SmoothingUNK((sent[i - 2], sent[i - 1], sent[i])))
                    for i in range(2, len(sent))]
        raise ValueError("model must be 'unigram', 'bigram' or 'trigram'")

    def sentenceLogProb(self, sent, model="bigram"):
        sent = list(sent)
        if len(sent) == 1:
            return math.log(self.unigramProb_SmoothingUNK(sent[0]))
        return sum(self._log_terms(sent, model))

    def sentenceProb(self, sent, model="bigram"):
        """Smoothed probability of a token list, e.g. ['<s>', 'bu', 'bir', 'cümle', '.', '</s>']."""
        return math.exp(self.sentenceLogProb(sent, model))

    def perplexity(self, testFile, model="bigram"):
        """exp(-1/N * sum log P(w_i | context)) over every scored token of the test file."""
        log_sum, n = 0.0, 0
        for sent in tokenize_file(testFile):
            terms = self._log_terms(sent, model)
            log_sum += sum(terms)
            n += len(terms)
        return math.exp(-log_sum / n) if n else float("inf")

    # -------------------------------------------------------------- generation
    def generateSentence(self, sent=["<s>"], maxFollowWords=3, maxWordsInSent=20):
        """Sample the next word from the top-k bigram followers, weighted by bigram counts."""
        sent = list(sent)
        while len(sent) < maxWordsInSent and sent[-1] != EOS:
            candidates = self._followers.get(sent[-1], [])[:maxFollowWords]
            if not candidates:
                break
            words, counts = zip(*candidates)
            sent.append(random.choices(words, weights=counts, k=1)[0])
        if sent[-1] != EOS:
            sent.append(EOS)
        return sent


def main():
    parser = argparse.ArgumentParser(description="Train an n-gram LM and evaluate perplexity.")
    parser.add_argument("--train", default="train_corpus.txt")
    parser.add_argument("--test", default="visible_testCorpus.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    lm = ngramLM()
    lm.trainFromFile(args.train)
    print(f"Training corpus: {args.train}")
    print(f"  tokens: {lm.numOfTokens}  vocabulary: {lm.sizeOfVocab}  sentences: {lm.numOfSentences}")

    print(f"\nPerplexity on {args.test} (add-1 smoothing):")
    for model in ("unigram", "bigram", "trigram"):
        print(f"  {model:8s} {lm.perplexity(args.test, model):8.2f}")

    print("\nTop 10 bigrams:")
    for (w1, w2), c in lm.bigrams()[:10]:
        print(f"  {w1} {w2}: {c}")
    print("\nTop 10 trigrams:")
    for (w1, w2, w3), c in lm.trigrams()[:10]:
        print(f"  {w1} {w2} {w3}: {c}")

    print(f"\nGenerated sentences (top-3 followers, seed {args.seed}):")
    for _ in range(3):
        print("  " + " ".join(lm.generateSentence(["<s>"], 3, 20)))


if __name__ == "__main__":
    main()
