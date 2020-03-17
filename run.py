import nltk
import numpy as np 
import pandas as pd
import csv
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import PunktSentenceTokenizer

# Tokenize and POS
# data = pd.read_csv('full_text.csv', sep=';')
# x = data.loc[:,'abstract']
# tokens = [nltk.word_tokenize(sentence) for sentence in x]
# print(tokens)

# data = pd.read_csv('full_text.csv', sep=';')
# a = data.loc[:, 'abstract']
# sentence=''.join(a)
# b = nltk.word_tokenize(sentence)
# print(len(b))

data = pd.read_csv('full_text.csv', sep=';')
x = data.loc[:,'abstract']
tokens=x.apply(lambda sentence: nltk.word_tokenize(sentence))
# tokens.to_csv('tokens.csv')

df = pd.read_csv('tokens.csv')
print(df)

# # Adjust tags
# def penn_to_wn(tag):

#     if tag.startswith('J'):
#         return wn.ADJ
#     elif tag.startswith('N'):
#         return wn.NOUN
#     elif tag.startswith('R'):
#         return wn.ADV
#     elif tag.startswith('V'):
#         return wn.VERB
#     return None

# lemmatizer = WordNetLemmatizer()

# # Get scores
# def get_sentiment(word,tag):

#     wn_tag = penn_to_wn(tag)
#     if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
#         return []

#     lemma = lemmatizer.lemmatize(word, pos=wn_tag)
#     if not lemma:
#         return []

#     synsets = wn.synsets(word, pos=wn_tag)
#     if not synsets:
#         return []

#     synset = synsets[0]
#     swn_synset = swn.senti_synset(synset.name())

#     return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

# # print scores
# ps = PorterStemmer()
# words_data = []
# pos_val = nltk.pos_tag(tokens)
# senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
# # print(f"pos_val is {pos_val}")
# # print(f"senti_val is {senti_val}")

# # format as dataframe
# data = senti_val
# df = pd.DataFrame(data)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)

# # sum and input to equation 


 