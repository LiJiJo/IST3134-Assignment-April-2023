#!/usr/bin/env python3
import sys
# Import necessary libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import string
import emoji
import csv

# Load English stopwords
stop_words = set(stopwords.words('english'))
def remove_noise(content_tokens, stop_words):
    cleaned_tokens=[]
    for token in content_tokens:
        token = re.sub('http([!-~]+)?','',token)
        token = re.sub('//t.co/[A-Za-z0-9]+','',token)
        token = re.sub('(@[A-Za-z0-9_]+)','',token)
        token = re.sub('[0-9]','',token)
        token = re.sub('[^ -~]','',token)
        token = emoji.replace_emoji(token, replace='')
        token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
        token = re.sub('[^\x00-\x7f]','', token) 
        token = re.sub(r"\s\s+" , " ", token)
        if (len(token)>3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def lemmatize_sentence(token):
    # initiate wordnetlemmatizer()
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence=[]
    
    # each of the words in the doc will be assigned to a grammatical category
    # part of speech tagging
    # NN for noun, VB for verb, adjective, preposition, etc
    # lemmatizer can determine role of word 
        # and then correctly identify the most suitable root form of the word
    # return the root form of the word
    for word, tag in pos_tag(token):
        if tag.startswith('NN'):
            pos='n'
        elif tag.startswith('VB'):
            pos='v'
        else:
            pos='a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
    return lemmatized_sentence
# Preprocessing function to tokenize, lemmatize, and remove noise
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemma_tokens = lemmatize_sentence(tokens)
    cleaned_tokens = remove_noise(lemma_tokens, stop_words)
    return cleaned_tokens

is_first_line = True  # Initialize a flag to track the first line

for line_number, line in enumerate(sys.stdin, start=0):
    if is_first_line:
        is_first_line = False
        continue
    line = line.strip()
    # Split the CSV line into individual columns
    # columns = line.split(',')
    csv_reader = csv.reader([line])
    columns = next(csv_reader)
    #Hopefully, 9th column is consistently about the abstract
    abstract = columns[8]  

    # Apply text preprocessing to the abstract
    cleaned_tokens = preprocess_text(abstract)
    if not cleaned_tokens:
        continue
    # Emit key-value pair where key is the index and value is a list of cleaned tokens
    print(f"{line_number}\t{cleaned_tokens}")
