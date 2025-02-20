import numpy as numpy
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def pre_processtext(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = pre_processtext(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
st.title('Sentiment Analysis')
st.write('Enter a statement')

user_input = st.text_area('Type here')

if st.button('Submit'):
    preprossed_input = pre_processtext(user_input)
    prediction = model.predict(preprossed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'sentiment: {sentiment}')
    st.write(f'score: {prediction}')

else:
    st.write('Please enter the text')


