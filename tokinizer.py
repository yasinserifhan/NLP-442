import sys
from collections import Counter


def train_BPE(file_name, init_vocab, max_merge_count=10, topK=1):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
        print("File loaded successfully!")
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")


    vocab = init_vocab.copy()
    merges = []

    # chaning chars to tokinzied version for better merge handeling 
    stoi = {char:idx for idx, char in enumerate(init_vocab)}
    itos = {idx:char for idx, char in enumerate(init_vocab)}

    encoder = lambda x: [stoi[c] for c in x]
    decoder = lambda x: itos[x]

  
    # first divisin part with the words 
    textList = division(text)

    # encoding the list of the words to 
    # representing version in our encoder 
    for i in range(len(textList)):
        textList[i] = encoder(textList[i]) 
    


    tupleList = counterTupleFirst(textList, decoder)
    

    for i in range(max_merge_count):
        selectedTuple = selection(tupleList, decoder, topK)
        merges.append((decoder(selectedTuple[0]), decoder(selectedTuple[1]), selectedTuple[2]))

        vocabUpdate(vocab, itos, stoi, selectedTuple, decoder)
        tupleList = update(textList, selectedTuple, decoder, vocab)

    return merges, vocab   



def test_BPE(file_name, merges, vocabulary):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
        print("File loaded successfully!")
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    
    vocab = vocabulary.copy()
    
    stoi = {char:idx for idx, char in enumerate(vocab)}
    itos = {idx:char for idx, char in enumerate(vocab)}

    encoder = lambda x: [stoi[c] for c in x]
    

    trainList = division(text)
    tokenizedLists = tokenization(trainList, merges)


    tokenized_corpus = []
    for i in range(len(tokenizedLists)):
        tokenized_corpus += tokenizedLists[i]
    
    input_ids = encoder(tokenized_corpus)
    
    
    
  
    return tokenized_corpus, input_ids


def print_truncated(lst, file=sys.stdout):
    if len(lst) > 100:
        print(f"{lst[:50]} ... {lst[-50:]}", file=file)
    else:
        print(lst, file=file)

def division(text):
    """
    Function to divide text in _word_ format.

    Args:
        text (str): The text to be divided.

    """
    return ["_" + word + "_" for word in text.split()]


    
def counterTupleFirst(textList, de):
    """
    Calculates the frequency of adjacent subword pairs in a given text list 
    and returns them as a sorted list of tuples.

    Args:
        textList (list of lists): The list of divided text (usually subwords) 
            to be processed.
        de (callable): A decoder function used to merge the pieces and 
            convert them back into a normal string format.

    Returns:
        list of tuples: A sorted list containing the processed pairs and their 
            frequencies. Format: [(part1, part2, frequency), ...].
            The sorting priority is as follows:
            1. Descending order of frequency (highest to lowest),
            2. Ascending order of the decoded string's length,
            3. Alphabetical order of the decoded string.
    """
    totalCounts = []

    for word in textList:

        totalCounts.extend([(word[i -1], word[i]) for i in range(1, len(word))])

    counts = Counter(totalCounts)

    counterTuple = [(key[0], key[1], counts[key]) for key in counts.keys()]
    counterTuple = sorted(counterTuple, key=lambda x: (-x[2], len(de(x[0]) + de(x[1])), de(x[0]) + de(x[1])))
    return counterTuple

def selection(tupleList, de, topK=1):
    """
    Selecting Best candidate for in first topk values by looking 
        1. starting with _
        2. ending with _
        3. if neither 1 nor 2 select first element from list

    Args:
        tupleList (List of tuple): sorted list of subwords pair tuples,
        de(callable): A decoder function for turning subwords index to string,
        topK(integer): The number of the total candidates to evalute. Default is 1

    Returns:
        tuple: Give the subword which most suitable to conditions.
    """
    # TODO: de = decoder kısmını ekle buna ve first tuple train içerisine koyarken
    candidates =  tupleList[:topK]

    for i in range(len(candidates)):
        if de(candidates[i][0])[0] == "_" :
            return candidates[i]

    for i in range(len(candidates)):
        if de(candidates[i][1])[-1] == "_":
            return candidates[i]

    return candidates[0]


def vocabUpdate(vocab, itos, stoi, selectedTuple, de):
    """
    Updating all the subword vocablary with the selectedtuple 
    and adding the new Subword to encoder and decoder functions

    Args:
        vocab(list): List of the subwords we using
        itos(dictionary): Decoder for turning Integer to string.(integer to String)
        stoi(dictionary): Encoder for turning String to Integer.(String to integer)
        SelectedTuple(Tuple): Tuple which contains the next word
        de(callable): Decoder function for getting subwords to string format using itos for the vocablary
    
    """
    subword = de(selectedTuple[0]) + de( selectedTuple[1])
    vocab.append(subword)
    stoi |=  {subword: len(vocab)-1}
    itos |= {len(vocab)-1: subword}


def update(textList, selectedTuple, de, vocab):
    """
    Updating all the textList according
    to new subword and new adjacent pairs  
    
    Args:
        textList (list of lists): The list of divided text (usually subwords) 
        to be processed.
        SelectedTuple(Tuple): Tuple which contains the next word
        de(callable): Decoder function for getting subwords to string format using itos for the vocablary
        vocab(list): List of the subwords we using

    Returns:
        counterTupleFirst(callable): Counter of the encoded textList for the adjacent pairs in the tuple format 
    """    
    for k in range(len(textList)):

        encodedWords = textList[k]
        new_id = len(vocab) -1

        newEncoded = []

        i=0
        while i<len(encodedWords):
            #checking the merged subwords first 
            if i < len(encodedWords) - 1 and \
                encodedWords[i] == selectedTuple[0] and \
                encodedWords[i+1] == selectedTuple[1]:

                newEncoded.append(new_id)
                i += 2
            
            else:
                newEncoded.append(encodedWords[i])
                i += 1
        
        textList[k] = newEncoded

    
    return counterTupleFirst(textList, de)
                
def tokenization(trainList, mergeList):
    """
    Applies a list of learned subword merges to a tokenized dataset.

    This function iterates through prioritized subword pairs and systematically 
    combines them within the provided text list. It effectively reconstructs 
    larger subwords or full words based on the BPE (Byte-Pair Encoding) rules.

    Args:
        trainList (list of lists): The dataset containing tokenized words 
            (lists of characters or subwords). Modified in place.
        mergeList (list of tuples/lists): A list of pairs to be merged, 
            ordered by priority. Format: [(part1, part2), ...].

    Returns:
        list of lists: The updated `trainList` where adjacent subwords matching 
            the merge rules have been combined into single tokens.
    """
    for merge in mergeList:
        merge = merge[0] + merge[1]
        for i in range(len(trainList)):
            word = trainList[i]
            new_word = []

            j=0
            while j < len(word)-1:
                subword = word[j] + word[j+1]
                if subword == merge:
                    new_word.append(subword)
                    j +=2
                
                else:
                    new_word.append(word[j])
                    j +=1
            if not j >= len(word):
                new_word.append(word[-1])
            trainList[i] = new_word



    # TODO: gredy search
    # for j in range(len(trainList)):
    #     word = trainList[j]
    #     longest = len(max(vocab, key=len))
        
    #     #  For every word we getting largest
    #     #  subset with starting in that index
    #     new_Word = []
    #     # for every word in train 
    #     i=0
        
    #     while i < len(word):
    #         # the largest subword we can get
    #         for k in range(longest):
    #             candidate = word[i:i +longest-k]
                
    #             if candidate in vocab:
    #                 new_Word.append(candidate)
    #                 i += longest-k
    #                 break            

        
    #     trainList[j] = new_Word

    return trainList





    
    

if __name__ == "__main__":
    init_vocab = list(
        "abcçdefgğhıijklmnoöprsştuüvyzqwx"
        "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX"
        "0123456789"
        ".,;:!?\"'()[]{}<>-_=/+*%<>^~@#$&|\\`"
    )
    init_vocab.sort()
    



    
    # ------------------------------------------------------------
    # You can use the following configurations to test your code:
    # ------------------------------------------------------------

    configs = [
        ("train.txt", "test.txt", 10, 1, "myoutput.txt"),
        ("train.txt", "test.txt", 20, 3, "myoutput.txt"),
        ("train1.txt", "test1.txt", 250, 1, "myoutput1.txt"),
        ("train1.txt", "test1.txt", 250, 5, "myoutput1.txt"),
        ("train1.txt", "test1.txt", 250, 10, "myoutput1.txt"),
        ("train2.txt", "test2.txt", 250, 1, "myoutput2.txt"),
        ("train2.txt", "test2.txt", 250, 5, "myoutput2.txt"),
        ("train2.txt", "test2.txt", 250, 10, "myoutput2.txt"),
        # ... add your own configurations here
    ]

    # Initialize or clear the output files
    for filename in ["myoutput.txt", "myoutput1.txt", "myoutput2.txt"]:
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            pass

    for i, (train_file, test_file, max_merges, top_k, out_file) in enumerate(
        configs, 1
    ):
        print(f"Running Configuration {i}: {train_file} -> {out_file}...")

        merges, vocab = train_BPE(
            train_file, init_vocab, max_merge_count=max_merges, topK=top_k
        )
        tokenized, ids = test_BPE(test_file, merges, vocab)

        # Write to consolidated file in append mode with cross-platform compatibility
        with open(out_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(
                f"----- Configuration {i}: Testing with {train_file} and {test_file}: maxMerge={max_merges}, topK={top_k} -----\n"
            )
            print("Merge List:", file=f)
            print_truncated(merges, file=f)
            print("\nUpdated Vocabulary:", file=f)
            print_truncated(vocab, file=f)
            print("\nTokenized Corpus:", file=f)
            print_truncated(tokenized, file=f)
            print("\nInput IDs:", file=f)
            print_truncated(ids, file=f)
            print(f"\nToken Count: {len(tokenized)}", file=f)
            f.write("\n" + "=" * 50 + "\n\n")

# ------------------------------------------------------------------------------------------------------------------------




