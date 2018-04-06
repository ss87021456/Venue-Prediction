from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv, argparse

# python create_data_example.py --train ../input/cleaned_training.txt --validation ../input/cleaned_validation.txt

def run(train, validation, dim=50, num=20):
    print("Creating example dataformat bag-of-word dim: {0}, number of lines: {1} \n".format(dim, num))
    print("Loading dataset...")
    training = pd.read_csv(train, sep='\t', names=['Paper_Id','Paper_title','Publication_venue', \
                                                'Cited_Papers', 'Cited_Papers_Venues'], quoting=csv.QUOTE_NONE)
    validation = pd.read_csv(validation, sep='\t', names=['Paper_Id','Paper_title','Publication_venue', \
                                                'Cited_Papers', 'Cited_Papers_Venues'], quoting=csv.QUOTE_NONE)

    print("Cleaning up titles with nan value...")
    training = training[training['Paper_title'].notnull()]
    validation = validation[validation['Paper_title'].notnull()]

    clean_train_titles = training['Paper_title'].values
    clean_vali_titles = validation['Paper_title'].values

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    print("Creating bag-of-word vectorizer...")
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = dim)

    train_features = vectorizer.fit_transform(clean_train_titles)
    vali_features = vectorizer.fit_transform(clean_vali_titles)
    print("training dataset shape:", train_features.shape)
    print("validaiton dataset shape:", vali_features.shape)

    print("Creating Label Encoder..")
    train_labels = training['Publication_venue']
    vali_labels = validation['Publication_venue']

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    vali_labels = le.transform(vali_labels)


    # output subset dataset example
    print("Outputing subset dataset example...")
    train_features = [",".join(str(w) for w in x) for x in train_features.toarray()[:num]]
    vali_features = [",".join(str(w) for w in x) for x in vali_features.toarray()[:num]]

    train_output = {"Encoded_Label":train_labels[:num], "Feature_Vector":np.array(train_features)}
    vali_output = { "Encoded_Label":vali_labels[:num], "Feature_Vector":np.array(vali_features)}
    col = ['Feature_Vector', 'Encoded_Label']
    t_output = pd.DataFrame(data=train_output)
    t_output = t_output[col]
    v_output = pd.DataFrame(data=vali_output)
    v_output = v_output[col]
    print(t_output.head())

    print("Saving ../input/text_train_features.txt ...")
    t_output.to_csv("../input/text_train_features.txt", sep='\t', index=False, header=False)
    
    print("Saving ../input/text_vali_features.txt ...")
    v_output.to_csv("../input/text_vali_features.txt", sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Input training dataset filename.")
    parser.add_argument("--validation", help="Input validation dataset filename.")
    parser.add_argument("--dim", default=50, type=int,  help="bag-of-word dim.")
    parser.add_argument("--num", default=20, type=int, help="output example lines number.")
    
    args = parser.parse_args()

    run(args.train, args.validation, args.dim, args.num)