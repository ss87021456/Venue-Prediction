import re, sys, csv, argparse
import nltk
import pandas as pd
import numpy as np
import pickle as pkl
from util import pre_clean, cleaned

# python clean_data.py --input ../input/training.txt --output ../input/cleaned_training.txt

def run(input, output, f):
    dataset = pd.read_csv(input, sep='\t', names=['Paper_Id','Paper_title','Publication_venue', \
                                                     'Cited_Papers', 'Cited_Papers_Venues'], quoting=csv.QUOTE_NONE)
    titles = dataset['Paper_title'].values
    
    pre_clean_sentences, word_list = pre_clean(titles)
    print("Done 1.lowercase 2.Remove criteria 3.Tokenize...")
    
    try:
        print("Loading pre-define",f,"...")
        filter_word = pkl.load(open(f, 'rb'))
    except:
        print("Didn't load pre-define, generate new...")
        freq = nltk.FreqDist(word_list)
        filter_word = list(filter(lambda x: x[1]>=5,freq.items())) # filter out low freq word
        filter_word = set([x for x, _ in filter_word])
        pkl.dump(filter_word, open("filter_word.pkl", 'wb'))
        print("From {0} words reduce to {1} words by filtering freq out < 5.".format(len(freq), len(filter_word)))

    cleaned_sentences = cleaned(pre_clean_sentences, filter_word)
    print("Done 4.filter out low freq...")
    
    dataset['Paper_title'] = cleaned_sentences
    print("Saving",output,"...")
    dataset.to_csv(output, sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input dataset filename.")
    parser.add_argument("--output", help="Output dataset filename.")
    parser.add_argument("--f", nargs='?', type=str, default="filter_word.pkl", 
                        help="Load filter_word.pkl")
    args = parser.parse_args()

    run(args.input, args.output, args.f)