import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

def extract_features(sentence, index):
    word = sentence[index]
    
    features = {
        # Basic features
        'word': word.lower(), # Word identity (lowercase is preferable) 
        'is_capitalized': word[0].isupper(), # Orthographic features 
        'is_numeric': word.isdigit(),
        #adding affix features for the word, which can help capture 
        # morphological patterns that are often indicative of certain
        #  POS tags. For example, words ending in "-ing" are often verbs
        # , while those ending in "-ly" are often adverbs.
        'prefix-2': word[:2].lower(), # Affixes
        'suffix-2': word[-2:].lower(),
        'prev_word': sentence[index - 1].lower() if index > 0 else '<START>', # Contextual features
        'next_word': sentence[index + 1].lower() if index < len(sentence) - 1 else '<END>'
    }
    
    return features

def main():
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")
    
    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)
    
    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")

    X_train = []
    y_train = []


    # 1. Training data: one feature dict per token, with its gold tag as the label.
    for sent in train_sents:
        words = [w for w, t in sent]
        tags = [t for w, t in sent]
        
        for i in range(len(words)):
            features = extract_features(words, i)
            X_train.append(features)
            y_train.append(tags[i])

    # 2. Gold tags for the test tokens.
    
    y_test = []
    for sent in test_sents:
        for word, tag in sent:
            y_test.append(tag)
    
    # Define the Pipeline 
    # The pipeline consists of a DictVectorizer to convert feature dictionaries into a sparse matrix format suitable for machine learning algorithms, and a LogisticRegression classifier to perform the actual tagging. 
    model = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Train the pipeline
    model.fit(X_train, y_train) 
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")

    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]
        
        # Extract features for the current sentence
        sent_features = [extract_features(words, i) for i in range(len(words))]
        pred_tags = model.predict(sent_features)
        pred_tags_list = [str(t) for t in pred_tags]
        
        y_pred.extend(pred_tags_list)
        
        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags_list))
        
        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")

if __name__ == "__main__":
    main()
