import gensim.downloader
import numpy as np

# Pretrained Google News word2vec (300-d). First use downloads ~1.6 GB via gensim.
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = model.vector_size


def replace_with_similar(sentence, indices):
    """
    Replaces the words at the given positions with their nearest word2vec neighbour.
    Returns the new sentence and {original_word: top-5 neighbours with cosine similarity}.
    """
    # Break the sentence into chunks, handling both strings and lists just in case
    word_pieces = sentence.split() if isinstance(sentence, str) else list(sentence)
    saved_matches = {}
    
    for pos in indices:
        # Make sure the index isn't out of bounds
        if 0 <= pos < len(word_pieces):
            curr_w = word_pieces[pos]
            
            # Only try to replace it if the model actually knows the word
            if curr_w in model:
                # Keep the top-5 neighbours for reporting
                top_guesses = model.most_similar(curr_w, topn=5)
                
                if top_guesses:
                    # Save all 5 guesses to our dictionary tracker
                    saved_matches[curr_w] = top_guesses
                    # But only physically replace the word with the #1 best guess
                    word_pieces[pos] = top_guesses[0][0]
                    
    # Stitch it all back together into a normal string
    rebuilt_text = " ".join(word_pieces)
    return rebuilt_text, saved_matches


def sentence_vector(sentence):
    """
    Turns a whole sentence into a single 300-dimensional math vector.
    It does this by grabbing the vectors for each word and finding the average.
    """
    word_pieces = sentence.split() if isinstance(sentence, str) else list(sentence)
    vec_tracker = {}
    
    for w in word_pieces:
        if w in model:
            vec_tracker[w] = model[w]
        else:
            # If the model has never seen this word before, just use a blank array of zeros
            vec_tracker[w] = np.zeros(EMBEDDING_DIM)
            
    # If the sentence was totally empty or gibberish, return a safe zero array
    if not vec_tracker:
        avg_sent_vec = np.zeros(EMBEDDING_DIM)
    else:
        # Calculate the average across all the unique word vectors we found
        avg_sent_vec = np.mean(list(vec_tracker.values()), axis=0)
        
    return vec_tracker, avg_sent_vec


def most_similar_sentences(file_path, query):
    """
    Reads a file full of sentences and figures out which ones are closest 
    in meaning to our search query using the cosine similarity math formula.
    """
    # Grab the vector for our search query (we can ignore the dictionary it returns)
    _, q_vec = sentence_vector(query)
    sentence_scores = []
    
    # Helper math function to figure out the angle/distance between two vectors
    def get_sim_math(vec_a, vec_b):
        bottom_a, bottom_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        # Avoid division by zero
        if bottom_a == 0 or bottom_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (bottom_a * bottom_b)

    # Open up the text file and read it line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            curr_line = line.strip()
            # Skip empty lines
            if not curr_line:
                continue
                
            # Get the vector for this specific line and compare it to our search query
            _, line_vec = sentence_vector(curr_line)
            closeness = get_sim_math(q_vec, line_vec)
            sentence_scores.append((curr_line, closeness))
            
    # Sort them so the highest scores (best matches) are at the top of the list
    sentence_scores.sort(key=lambda item: item[1], reverse=True)
    return sentence_scores


def analyze_dimension_contributions(word1, word2, top_k=10):
    """
    Looks under the hood to see exactly which dimensions inside the 300-D arrays
    are pushing these two words together (or pulling them apart).
    """
    # If either word is missing from the dictionary, bail out
    if word1 not in model or word2 not in model:
        return []
        
    val1 = model[word1]
    val2 = model[word2]
    
    # Multiply them together element-by-element to see the impact of each spot
    dim_impacts = val1 * val2
    
    # We need to look at absolute values here because big negative numbers 
    # are just as important as big positive ones! Sort them to find the strongest.
    strongest_spots = np.argsort(np.abs(dim_impacts))[-top_k:][::-1]
    
    final_stats = []
    for spot in strongest_spots:
        spot_idx = int(spot)
        # (dimension index, product, value in word1, value in word2)
        final_stats.append((
            spot_idx, 
            float(dim_impacts[spot_idx]), 
            float(val1[spot_idx]), 
            float(val2[spot_idx])
        ))
        
    return final_stats