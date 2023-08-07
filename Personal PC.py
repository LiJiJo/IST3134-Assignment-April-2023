from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import string
import emoji
import numpy as np
import pandas as pd

stop_words=stopwords.words('english')
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

# Load the new dataset and access the 'content' column
# cdf=combined_df
cdf = pd.read_csv('metadata.csv', low_memory=False)
content_column = cdf['abstract']

content_column=content_column.drop_duplicates()
content_column=content_column.dropna()

# Preprocess the text data
import time

# Assuming preprocess_text and content_column are defined properly
start_time = time.time()
cleaned_tokens = [preprocess_text(text) for text in content_column]
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

    
import csv
import json
# Specify the filename for the CSV file
filename = "cord19_cleaned_abstract.csv"

# # Writing the nested list to the CSV file
with open(filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(cleaned_tokens)

cleaned_tokens = []
with open(filename, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        cleaned_tokens.append(row)

# # Writing the nested list to the JSON file
# filename = "cord19_cleaned_abstract.json"
# with open(filename, "w") as jsonfile:
#     json.dump(cleaned_tokens, jsonfile)


cleaned_sentences = [' '.join(tokens) for tokens in cleaned_tokens]
# Specify the filename for the CSV file
filename = "cord19_a_COMBINED.csv"

cleaned_sentences = [item for item in cleaned_sentences if item]
# Writing the list to the CSV file
# with open(filename, "w", newline="") as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(cleaned_sentences)
#     # for tokens in cleaned_sentences:
#     #     csv_writer.writerow(tokens)
with open(filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    for sentence in cleaned_sentences:
        csv_writer.writerow([sentence])

cleaned_sentences = []
with open(filename, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        cleaned_sentences.append(','.join(row))

cleaned_short_sentences=cleaned_sentences[:20]
filename = "cord19_a_short_sentence.csv"
with open(filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    for sentence in cleaned_short_sentences:
        csv_writer.writerow([sentence])
        
# Check the dataset
import re
count_non_alphanumeric=0
with open(filename, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        for value in row:
            if not value:
                print("Found empty value in the dataset.")
                count_non_alphanumeric += 1
            # elif re.fullmatch(r'[^a-zA-Z0-9]*', value):
            #     print("Found non-alphanumeric character-only value:", value)
            #     count_non_alphanumeric += 1


print("Total non-alphanumeric character-only values found:", count_non_alphanumeric)

# class TopicModelingMR(MRJob):

#     def steps(self):
#         return [
#             MRStep(mapper=self.mapper,
#                    reducer=self.reducer),
#             MRStep(reducer=self.final_reducer)
#         ]

#     def remove_noise(self, content_tokens, stop_words):
#         cleaned_tokens=[]
#         for token in content_tokens:
#             token = re.sub('http([!-~]+)?','',token)
#             token = re.sub('//t.co/[A-Za-z0-9]+','',token)
#             token = re.sub('(@[A-Za-z0-9_]+)','',token)
#             token = re.sub('[0-9]','',token)
#             token = re.sub('[^ -~]','',token)
#             token = emoji.replace_emoji(token, replace='')
#             token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
#             token = re.sub('[^\x00-\x7f]','', token) 
#             token = re.sub(r"\s\s+" , " ", token)
#             if (len(token)>3) and (token not in string.punctuation) and (token.lower() not in stop_words):
#                 cleaned_tokens.append(token.lower())
#         return cleaned_tokens

#     def lemmatize_sentence(self, token):
#         # initiate wordnetlemmatizer()
#         lemmatizer = WordNetLemmatizer()
#         lemmatized_sentence=[]
        
#         # each of the words in the doc will be assigned to a grammatical category
#         # part of speech tagging
#         # NN for noun, VB for verb, adjective, preposition, etc
#         # lemmatizer can determine role of word 
#         # and then correctly identify the most suitable root form of the word
#         # return the root form of the word
#         for word, tag in pos_tag(token):
#             if tag.startswith('NN'):
#                 pos='n'
#             elif tag.startswith('VB'):
#                 pos='v'
#             else:
#                 pos='a'
#             lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
#         return lemmatized_sentence

#     def preprocess_text(self, text):
#         tokens = word_tokenize(text)
#         lemma_tokens = self.lemmatize_sentence(tokens)
#         stop_words_path = self.options.stop_words
#         stop_words = set(stopwords.words('english'))
#         if stop_words_path:
#             with open(stop_words_path) as f:
#                 custom_stop_words = f.read().splitlines()
#             stop_words.update(custom_stop_words)
#         cleaned_tokens = self.remove_noise(lemma_tokens, stop_words)
#         return cleaned_tokens
#     def mapper(self, _, line):
#         doc_id, abstract = line.strip().split('\t')
#         # Preprocess the abstract text
#         cleaned_tokens = self.preprocess_text(abstract)
#         yield None, cleaned_tokens

#     def combine_documents(self, _, cleaned_token_lists):
#         # Combine the cleaned tokens from all documents into one corpus
#         combined_corpus = []
#         for token_list in cleaned_token_lists:
#             combined_corpus.extend(token_list)
#         yield None, combined_corpus

#     # def final_reducer(self, _, doc_topic_lists):
#     #     # In the final reducer, you can perform any post-processing or analysis
#     #     # For example, you might sort the documents based on their relevance to each topic
#     #     for doc_id, topics in doc_topic_lists:
#     #         yield doc_id, topics
#     def final_reducer(self, _, combined_corpus):
#         # Create a dictionary from the combined corpus and convert it into a bag-of-words representation
#         dictionary = corpora.Dictionary(combined_corpus)
#         corpus = [dictionary.doc2bow(tokens) for tokens in combined_corpus]

#         # Perform LDA on the combined corpus
#         num_topics = 5  # You can adjust the number of topics as needed
#         lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

#         # Yield the topics and their corresponding word distributions
#         topics = []
#         for topic_id in range(num_topics):
#             topic_words = lda_model.show_topic(topic_id)
#             topic_words = [(word, prob) for word, prob in topic_words]
#             topics.append(topic_words)
#             # Calculate coherence score (using u_mass)
#             coherence_scores = []
#             for topic_id, topic in topic_info.items():
#                 dictionary = corpora.Dictionary([topic['keywords']])
#                 topic_corpus = [dictionary.doc2bow(topic['keywords'])]
#                 coherence_model = CoherenceModel(
#                     model=LdaModel(topic_corpus, id2word=dictionary, num_topics=1),
#                     texts=[topic['keywords']],
#                     dictionary=dictionary,
#                     coherence='u_mass'
#                 )
#                 coherence_score = coherence_model.get_coherence()
#                 topic_info[topic_id]['coherence_score'] = coherence_score
#                 coherence_scores.append((topic_id, coherence_score))
    
#             # Sort topics by coherence score in descending order
#             coherence_scores.sort(key=lambda x: x[1], reverse=True)
    
#             # Output topics and their information to separate files
#             for topic_id, coherence_score in coherence_scores:
#                 with open(f'topic_{topic_id}_keywords.txt', 'w') as f_keywords:
#                     f_keywords.write('\n'.join(topic_info[topic_id]['keywords']))
    
#                 with open(f'topic_{topic_id}_coherence_score.txt', 'w') as f_coherence:
#                     f_coherence.write(str(coherence_score))
    
#                 with open(f'topic_{topic_id}_num_records.txt', 'w') as f_records:
#                     f_records.write(str(len(topic_info[topic_id]['records'])))
    
#             # Also, you can print the information here if needed
#             for topic_id, coherence_score in coherence_scores:
#                 print(f"Topic {topic_id} - Coherence Score: {coherence_score}")
#                 print("Top 15 Keywords:")
#                 for keyword in topic_info[topic_id]['keywords'][:15]:
#                     print(keyword)
#                 print(f"Number of Records: {len(topic_info[topic_id]['records'])}")
#                 print("\n")


# if __name__ == '__main__':
#     TopicModelingMR.run()
