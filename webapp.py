import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import unicodedata
from tensorflow.keras.callbacks import TensorBoard
import datetime
import pickle

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Title
st.title("Toxic Comment Classification")


# load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('saved_model/model3.h5')
    return model


with st.spinner("Loading Model...."):
    model = load_model()


classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def rem_accented(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def rem_puntucation(text):
    pat = r'[^a-zA-Z0-9;\"\'\s]'
    return re.sub(pat, '', text)


def get_long(text):
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'s", " is ", text)
    return text


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!"?/:;\"\'\s]'
    return re.sub(pat, '', text)


# stopwords
stops = stopwords.words("english")
delete = ['no',
          'nor',
          'not',
          'don',
          "don't",
          'ain',
          'aren',
          "aren't",
          'couldn',
          "couldn't",
          'didn',
          "didn't",
          'doesn',
          "doesn't",
          'hadn',
          "hadn't",
          'hasn',
          "hasn't",
          'haven',
          "haven't",
          'isn',
          "isn't",
          'ma',
          'mightn',
          "mightn't",
          'mustn',
          "mustn't",
          'needn',
          "needn't",
          'shan',
          "shan't",
          'shouldn',
          "shouldn't",
          'wasn',
          "wasn't",
          'weren',
          "weren't",
          'won',
          "won't",
          'wouldn',
          "wouldn't"]
new = [stop for stop in stops if stop not in delete]

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def get_lemmatization(text):
    # Tokenize: Split the sentence into words
    word_list = nltk.word_tokenize(text)
    # Lemmatize list of words and join
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output
def preprocess(df):
    df['comment_text'] = df['comment_text'].astype(str)
    df['clean_comment'] = df['comment_text'].str.lower()
    df['clean_comment'] = df['clean_comment'].apply(rem_puntucation)
    df['clean_comment'] = df['clean_comment'].apply(get_long)
    df['clean_comment'] = df['clean_comment'].apply(remove_special_characters)
    df['clean_comment'] = df['clean_comment'].apply(lambda x: ' '.join([item for item in x.split() if item not in new]))
    df['clean_comment'] = df['clean_comment'].apply(rem_accented)
    df['clean_comment'] = df['clean_comment'].apply(get_lemmatization)
    return df
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# image preprocessing
def load(text):
    input1 = {'comment_text': text}  # input text here
    inputdf = pd.DataFrame(input1, index=[0])
    inputdf = preprocess(inputdf)
    X_tokenized_input = tokenizer.texts_to_sequences(inputdf['clean_comment'])
    X_ip = pad_sequences(X_tokenized_input, maxlen=200)
    return X_ip



textinput = st.text_input("Enter Text to classify...","")

# Get image from URL and predict
st.write("Predicting Class...")
with st.spinner("Classifying..."):
    X_ip = load(textinput)
    y_pred = model.predict(X_ip)
for i in range(0,6):
    st.write(classes[i], ':' , round(y_pred[0][i]*100,2),'%')