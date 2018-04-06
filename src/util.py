import re
import numpy as np

def pre_clean(titles, w_list=True):
    pre_clean_sentences = []
    word_list = [] 
    total_len = len(titles)
    count = 0
    for title in titles:
        if count % 10000 == 0:
            print("Cleaning step 1,2,3 {0} / {1}".format(count, total_len))
        count += 1
        title = title.lower() # lowercase
        #print(count, title)
        title = re.sub("[^a-zA-Z\- ]", " ", title) # Remove all characters not a-z, A-Z, hyphen, whitespace
        #print(count, title)
        pre_clean_sentences.append(title)
        if w_list:
            words = title.split(" ") # Tokenize each title into words by splitting on whitespace.
            word_list += words
    if w_list:
        return np.array(pre_clean_sentences), word_list
    else:
        return np.array(pre_clean_sentences)

def cleaned(pre_clean_sentences, filter_word):
    cleaned_sentences = []
    total_len = len(pre_clean_sentences)
    count = 0
    for sentence in pre_clean_sentences:
        if count % 10000 == 0:
            print("Cleaning step 4 {0} / {1}".format(count, total_len))
        count += 1
        words = sentence.split(" ")
        words = [x for x in words if x in filter_word]
        cleaned_sentences.append(" ".join(words))
    return np.array(cleaned_sentences)