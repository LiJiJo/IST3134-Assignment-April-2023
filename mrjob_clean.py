from mrjob.job import MRJob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import string
import emoji
import csv

import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Load English stopwords
stop_words = set(stopwords.words('english'))

# Define your preprocessing functions (remove_noise, lemmatize_sentence, preprocess_text) here
def remove_noise(content_tokens, stop_words):
    cleaned_tokens = []
    for token in content_tokens:
        token = re.sub('http([!-~]+)?', '', token)
        token = re.sub('//t.co/[A-Za-z0-9]+', '', token)
        token = re.sub('(@[A-Za-z0-9_]+)', '', token)
        token = re.sub('[0-9]', '', token)
        token = re.sub('[^ -~]', '', token)
        token = emoji.replace_emoji(token, replace='')
        token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()
        token = re.sub('[^\x00-\x7f]', '', token)
        token = re.sub(r"\s\s+", " ", token)
        if (len(token) > 3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def lemmatize_sentence(token):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(token):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def preprocess_text(text):
    tokens = word_tokenize(text)
    lemma_tokens = lemmatize_sentence(tokens)
    cleaned_tokens = remove_noise(lemma_tokens, stop_words)
    return cleaned_tokens

# class MRPreprocess(MRJob):
#     def mapper(self, _, line):
#         line = line.strip()
#         # Split the CSV line into individual columns
#         columns = line.split(',')

#         # Assuming the 9th column contains the 'abstract'
#         abstract = columns[9]  # Index 8 corresponds to the 9th column (0-based indexing)

#         # Apply text preprocessing to the abstract
#         cleaned_tokens = preprocess_text(abstract)

#         # Emit key-value pair where key is the index and value is a list of cleaned tokens
#         yield columns[0], cleaned_tokens

#     def reducer(self, key, values):
#         # values is an iterator containing lists of cleaned tokens for each abstract
#         for cleaned_tokens in values:
#             # Join the cleaned tokens with a space as the delimiter
#             cleaned_tokens_str = ' '.join(cleaned_tokens)
#             # Emit the line number and joined cleaned tokens as the output
#             yield key, cleaned_tokens_str

class MRJobClean(MRJob):

    def mapper(self, _, line):
        # Skip the first row (header)
        if self.line_number() == 0:
            return
        
        line = line.strip()
        csv_reader = csv.reader([line])
        columns = next(csv_reader)
        abstract = columns[9]
        
        # Apply text preprocessing to the abstract
        cleaned_tokens = preprocess_text(abstract)
        if not cleaned_tokens:
            return
        
        # Emit key-value pair where key is the index and value is a list of cleaned tokens
        yield str(self.line_number()), cleaned_tokens
        
    def reducer(self, key, values):
           current_tokens = []
           
           for value in values:
               current_tokens.extend(value)
    
           # Remove any empty or None tokens from the list
           current_tokens = [token for token in current_tokens if token]
    
           if current_tokens:
               # Join the cleaned tokens and enclose them in double quotes
               cleaned_tokens_str = '"' + ' '.join(current_tokens) + '"'
               
               yield key, cleaned_tokens_str

if __name__ == '__main__':
    MRJobClean.run()
