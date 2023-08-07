from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, concat_ws, col
from pyspark.sql.types import ArrayType, StringType
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re
import string
import emoji
import csv
import time

start_time = time.time()

# Create a SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Preprocessing") \
    .getOrCreate()
 

sc=spark.sparkContext 
# Initialize NLTK resources (download if not already present)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up NLTK resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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

# Load the input CSV file into a DataFrame
input_file = "/user/hadoop/repository/cord19file/metadata.csv" 
# Read the CSV file using Spark DataFrame API
df = spark.read.option("header", "true").option("multiline", "true").csv(input_file)
# abstract = df.select("_c8")
# Define a user-defined function (UDF) for preprocessing
preprocess_udf = udf(preprocess_text, ArrayType(StringType()))

# Apply the UDF to the abstract column
# preprocessed_abstract = abstract.withColumn("preprocessed_abstract", preprocess_udf(abstract))
preprocessed_abstract = df.withColumn("preprocessed_abstract", preprocess_udf(col("abstract")))
# Concatenate tokens within each row with a space separator
cleaned_sentences = preprocessed_abstract.withColumn("cleaned_sentences", concat_ws(" ", "preprocessed_abstract"))

# Drop rows with empty cleaned_sentences
cleaned_sentences = cleaned_sentences.dropna(subset=["cleaned_sentences"])

# Save the cleaned sentences as a text file
output_directory = '/user/hadoop/spark_cordclean'
cleaned_sentences.select("cleaned_sentences").write.text(output_directory)

# Stop the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Stop the SparkSession
spark.stop()
