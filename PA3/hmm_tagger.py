import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import defaultdict, Counter

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

class HMMTagger:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.emissions = defaultdict(Counter)
        self.tags = set()
        self.vocab = set()
        self.tag_counts = Counter()

    def train(self, tagged_sents):
        # Count tag transitions (including <START> -> first tag) and word emissions.
        for sentence in tagged_sents:
            prev_tag = None
            for word, tag in sentence:
                self.tags.add(tag)
                self.vocab.add(word)
                self.tag_counts[tag] += 1
                
                # IF prev_tag is not None, we have a transition from prev_tag to tag, so we increment that count.
                if prev_tag is not None:
                    self.transitions[prev_tag][tag] += 1
                # counting for <START> tag to the first tag of the sentence
                else:
                    self.transitions['<START>'][tag] += 1  # Start of sentence transition
                self.emissions[tag][word] += 1
                prev_tag = tag

    def get_transition_prob(self, prev_tag, curr_tag):
        total_transitions = sum(self.transitions[prev_tag].values())
        
        # Laplace smoothing with log probabilities to avoid underflow issues
        prob = (self.transitions[prev_tag][curr_tag] + 1) / (total_transitions + len(self.tags))
        return np.log(prob)


    def get_emission_prob(self, word, tag):
        # Known words: relative frequency (with a tiny epsilon so log() never sees 0).
        # Unknown words: a small, tag-dependent probability mass, as specified in the assignment.
        if word in self.vocab:
            prob = (self.emissions[tag][word] + 1e-10) / self.tag_counts[tag]
        else:
            prob = 1 / (self.tag_counts[tag] + len(self.vocab))
        return np.log(prob)
    
    def viterbi(self, sentence):
        # Implement the Viterbi algorithm to find the most likely sequence of tags for the given sentence.
        n = len(sentence)
        if n == 0:
            return []

        viterbi = defaultdict(dict)
        backpointers = defaultdict(dict)
        
        # for tie-breaking, we will sort the tags alphabetically and iterate in that order.
        sorted_tags = sorted(list(self.tags))

        # Base case (t=0)
        for tag in sorted_tags: 
            viterbi[0][tag] = self.get_emission_prob(sentence[0], tag) + self.get_transition_prob('<START>', tag)
            backpointers[0][tag] = '<START>'

        # Fill the Viterbi matrix
        # with -inf because of the log probabilities 
        for t in range(1, n):
            for curr_tag in sorted_tags:
                max_prob = float('-inf')
                best_prev_tag = None

                for prev_tag in sorted_tags:
                    prob = viterbi[t-1][prev_tag] + self.get_transition_prob(prev_tag, curr_tag) + self.get_emission_prob(sentence[t], curr_tag)
                    # '>' is used for tie-breaking, so if two tags have the same probability, the one that comes first alphabetically will be chosen.
                    if prob > max_prob: 
                        max_prob = prob
                        best_prev_tag = prev_tag

                viterbi[t][curr_tag] = max_prob
                backpointers[t][curr_tag] = best_prev_tag

        # Backtrack to find the most likely tag sequence
        best_path = []
        current_tag = max(viterbi[n-1].items(), key=lambda x: x[1])[0]
        best_path.append(current_tag)

        for t in range(n-1, 0, -1):
            current_tag = backpointers[t][current_tag]
            best_path.append(current_tag)

        return list(reversed(best_path))

def main():

    tagged_sents = list(brown.tagged_sents(tagset='universal'))
   
    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")
    
    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)
    
    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")
    
    
    tagger = HMMTagger()
    tagger.train(train_sents)

    transitions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.transitions.items())[:5]}
    emissions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.emissions.items())[:5]}

    print(f"\nTagger Transitions (first 5 sorted): {transitions_summary}")
    print(f"\nTagger Emissions (first 5 sorted): {emissions_summary}")
    print(f"\nTagger Tags (sorted): {sorted(list(tagger.tags))}")
    print(f"\nVocabulary size: {len(tagger.vocab)}")
    print(f"\nTagger Vocab (first 5 sorted): {sorted(list(tagger.vocab))[:5]}")
    print(f"\nTagger Tag Counts (sorted items): {dict(sorted(tagger.tag_counts.items()))}")

    print("\n=== VITERBI DECODING & EVALUATION ===")
    y_true = []
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")
    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]
        
        pred_tags = tagger.viterbi(words)
        
        y_true.extend(true_tags)
        y_pred.extend(pred_tags)
        
        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags))
        
        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")
    

if __name__ == "__main__":
    main()
