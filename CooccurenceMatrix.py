import numpy as np
import pandas as pd
import itertools 
 
def generate_co_occurrence_matrix(corpus,win):
    print(corpus)
    vocab = set(corpus)
    vocab = list(vocab)
    print(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}


    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
    
    i=0
    for i in range(0,len(vocab)):
 #  Searching for the positions of the context words.
      indices = [j for j, x in enumerate(corpus) if x == vocab[i]]

      for k in range(0,len(indices)): 
# specifying the size of the window
        if (indices[k]<win): 
          min =indices[k] 
        else: min = win
# Selecting the window and incrementing the count every time we see a word occurrence
        window=corpus[(-min+indices[k]):(indices[k])]+corpus[(indices[k]+1):(indices[k]+win+1)]
        for j in range(0,len(window)):
            rel_pos=vocab.index(window[j])
            co_occurrence_matrix[i][rel_pos] = co_occurrence_matrix[i][rel_pos]+1
    return co_occurrence_matrix, vocab_index

text_data=[["Mary", "had", "a" ,"Lamb"], ["His", "fleece", "was", "white"],[ "He", "is", "smart"]]

#text_data=[["He", "is" ,"not", "lazy"],["He", "is", "intelligent"],["He","is" ,"smart"]] 
 
# Create one list using many lists




data = list(itertools.chain.from_iterable(text_data))
matrix, vocab_index = generate_co_occurrence_matrix(data,3)
 
 
data_matrix = pd.DataFrame(matrix, index=vocab_index, columns=vocab_index)
print(data_matrix)