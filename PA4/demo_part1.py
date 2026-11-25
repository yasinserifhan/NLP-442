"""Part I demo: word substitution, sentence vectors, semantic search, dimension analysis.

Adapted from the sample script provided with the assignment. Writes results_part1.txt.
"""
from word_similarity import (
    replace_with_similar,
    sentence_vector,
    most_similar_sentences,
    analyze_dimension_contributions,
)

SEP = "-" * 80

with open("results_part1.txt", "w", encoding="utf-8") as f:
    sentence = "NLP is a fascinating field of study and I love learning about it"
    new_sentence, neighbours = replace_with_similar(sentence, [3, 4, 10])
    print("Original :", sentence, file=f)
    print("Replaced :", new_sentence, end="\n\n", file=f)
    for word, top5 in neighbours.items():
        print(f"{word}: " + ", ".join(f"{w} ({s:.3f})" for w, s in top5), file=f)
    print(SEP, file=f)

    vectors, sent_vec = sentence_vector("I am a student studying NLP at Hacettepe University")
    print("Words with vectors:", list(vectors.keys()), file=f)
    print("Sentence vector length:", len(sent_vec), file=f)
    print(SEP, file=f)

    for query in ["Is swimming a good sport ?",
                  "Does Turkey have good universities ?",
                  "What happened to your backpack ?"]:
        print(f"Query: {query}", file=f)
        for text, score in most_similar_sentences("data/sentences_sample.txt", query)[:3]:
            print(f"  {score:.5f} -> {text}", file=f)
        print(file=f)
    print(SEP, file=f)

    for w1, w2 in [("good", "great"), ("good", "bad"), ("car", "banana")]:
        print(f"{w1} / {w2}: top-5 dimensions (index, product, v1, v2)", file=f)
        for item in analyze_dimension_contributions(w1, w2, top_k=5):
            print("  ", tuple(round(x, 4) if isinstance(x, float) else x for x in item), file=f)
        print(file=f)
