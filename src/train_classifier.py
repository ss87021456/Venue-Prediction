from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import csv, argparse

# python train_classifier.py --train ../input/cleaned_training.txt --validation ../input/cleaned_validation.txt

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def run(train, validation):
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
                                 max_features = 3000)
    vectorizer = vectorizer.fit(clean_train_titles)
    train_features = vectorizer.transform(clean_train_titles)
    vali_features = vectorizer.transform(clean_vali_titles)
    print("training dataset shape:", train_features.shape)
    print("validaiton dataset shape:", vali_features.shape)

    H_vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 3000)

    print("Creating Heterogenous feature...") # concatenate Cited_Papers_Venues info into titles
    t_mid = np.array([' ' for _ in range(len(training))])
    v_mid = np.array([' ' for _ in range(len(validation))])

    H_train_features = np.core.defchararray.add(training['Paper_title'].values.astype(str), t_mid)
    H_train_features = np.core.defchararray.add(H_train_features, training['Cited_Papers_Venues'].values)
    H_vali_features = np.core.defchararray.add(validation['Paper_title'].values.astype(str), v_mid)
    H_vali_features = np.core.defchararray.add(H_vali_features, validation['Cited_Papers_Venues'].values)

    vectorizer = H_vectorizer.fit(H_train_features)
    H_train_features = H_vectorizer.transform(H_train_features)
    H_vali_features = H_vectorizer.transform(H_vali_features)
    
    print("Creating Label Encoder..")
    train_labels = training['Publication_venue']
    vali_labels = validation['Publication_venue']

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    vali_labels = le.transform(vali_labels)

    print("Start training for Origianl...")
    clf = linear_model.SGDClassifier()
    clf.fit(train_features, train_labels)
    print("Origianl Training accuracy:", clf.score(train_features, train_labels))

    y_pred = clf.predict(vali_features)
    f1_ma = f1_score(vali_labels, y_pred, average='macro') 
    f1_mi = f1_score(vali_labels, y_pred, average='micro') 
    print("Origianl f1_macro: {0} / f1_micro: {1}".format(f1_ma, f1_mi))

    print("Start training for Heterogenous...")
    H_clf = linear_model.SGDClassifier()
    clf.fit(H_train_features, train_labels)
    print("Heterogenous Training accuracy:", clf.score(H_train_features, train_labels))

    H_y_pred = clf.predict(H_vali_features)
    f1_ma = f1_score(vali_labels, H_y_pred, average='macro') 
    f1_mi = f1_score(vali_labels, H_y_pred, average='micro') 
    print("Heterogenous info f1_macro: {0} / f1_micro: {1}".format(f1_ma, f1_mi))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Input training dataset filename.")
    parser.add_argument("--validation", help="Input validation dataset filename.")
    args = parser.parse_args()

    run(args.train, args.validation)