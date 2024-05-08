
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Given Corpus with different documents
corpus = {1: 'i am student of computer engineering at the university of guilan',
          2: 'i am studying natural language processing right now'}

# Create a bag of words for each documents
BoW = []
# Split each document
for row in corpus:
    BoW.append(corpus[row].split(' '))
    Num_document = row  # use it for making sets
# print(BoW)
# print(Num_document)

# Remove any duplicate words
unique_Words = set(BoW[0]).union(set(BoW[1]))
# print(uniqueWord)

# Create Document-Word matrix
# Sort unique words for better visualization
unique_Words = sorted(unique_Words)
# dict.fromkeys(x=keys, y=value)
Doc1 = dict.fromkeys(unique_Words, 0)
# Count word occurrence in Doc
for word in BoW[0]:
    Doc1[word] += 1