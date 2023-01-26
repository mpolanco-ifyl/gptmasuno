import streamlit as st
import nltk
st.write('Please use the NLTK Downloader to obtain the resource:')
nltk.download('punkt')
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Create a text box
text_input = st.text_input('Enter your text here:')

# Tokenize the text
text_tokens = word_tokenize(text_input)

# Load the English language model
nlp = English()

# Create a list of stop words
stop_words = stopwords.words('english')

# Process the tokens
processed_text = []
for word in text_tokens:
    if word not in stop_words:
        doc = nlp(word)
        lemma = [token.lemma_ for token in doc]
        processed_text.append(lemma[0])

# Create a dataframe
df = pd.DataFrame(columns=['Text'])
df['Text'] = processed_text

# Create LIWC categories
categories = ['Verbs', 'Nouns', 'Adjectives', 'Personal Pronouns']

# Count the LIWC categories
for category in categories:
    df[category] = 0

# Count the LIWC categories
for index, row in df.iterrows():
    doc = nlp(row['Text'])
    if doc.pos_ == 'VERB':
        df.at[index, 'Verbs'] += 1
    elif doc.pos_ == 'NOUN':
        df.at[index, 'Nouns'] += 1
    elif doc.pos_ == 'ADJ':
        df.at[index, 'Adjectives'] += 1
    elif doc.pos_ == 'PRON':
        df.at[index, 'Personal Pronouns'] += 1

# Display the dataframe
st.dataframe(df)
