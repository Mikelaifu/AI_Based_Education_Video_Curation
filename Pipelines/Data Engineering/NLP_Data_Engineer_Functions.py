
import pandas as pd
import numpy as np
import glob
import os
import string 
import re
import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from textblob import TextBlob, Word, Blobber

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from pprint import pprint


# meta_texts = pd.read_csv("Targeted_Output/Phase1/0_raw_metadata.csv")
# example_txt = meta_texts['raw_text'][0]
# example_txt


# extend stop words
def stopwords_generator(added_stopwords_lst):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(added_stopwords_lst)
    return stop_words


# added_stopwords_lst = ['Source:', 'pinterestcom', 'peacocksukcom',
#                        'enwikipediaorg', 'natureconservationin']

# stop_words = stopwords_generator(added_stopwords_lst = added_stopwords_lst)

# first stage clean: manual text cleaning 
# manualaly remove certain patterns from the text
# normanized text cleaning and tokenization of the words
# lemmatize th words
# tokenized the sentences

import gensim


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def lemmatize_preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
            
    return result


def text_cleaning_sub(any_txt):
    
        # convert to lowercase
        #strip space from left and right
        any_txt = any_txt.lower().lstrip().rstrip()

        # remove punctuation
        any_txt = re.sub('[^a-zA-z]', " ", any_txt)

        # remove tags
        any_txt = re.sub("&lt;/?.*&gt;", " &;t;&gt; ", any_txt)

        # remove spcial character and digits
        res = re.sub("(\\d|\\W)+", " ", any_txt)
        
        res = res.strip()
        
        return res
        

# the below functiosn will help clean any textchunk data into 
# lammatzied word, word tokens, unique work tokens, sentences tokens, word token in each sentence token, lemmaztied words in each sentences tokens 
def text_cleaning(txt, stop_words):
    
    corpus_lem_words = [] # contians Lemmatzied words
    cleaned_words = [] # contians cleaned and tokenized workds
    tokens_sentences = []
    
    # tokenzied sentences
    tokens_sentences =  [text_cleaning_sub(t) for t in nltk.sent_tokenize(txt)]
    
    # tokenzied words from tokenzied sentences
    tokens_words_sentences = [nltk.word_tokenize(t) for t in tokens_sentences]

    # initiate the splitting text
    text_words = txt.split(" ")
    
    for wrds in text_words:
       
        wrds = text_cleaning_sub(any_txt = wrds)
        
        # remove the space
        wrds = wrds.replace(" ", "")
        
        
        # Stemming 
        wrds_split = wrds.split()

        # Stemming
        ps = PorterStemmer()

        # call stop words
#         stopword_lst = stopwords.words('english')

        if (wrds not in stop_words) and (wrds != '') and (len(wrds)>2) :

            cleaned_words.append(wrds)   

            # Lemmatization
            lem = WordNetLemmatizer()
            text_lem = [lem.lemmatize(word) for word in wrds_split]
            text_lem = " ".join(text_lem)

            corpus_lem_words.append(text_lem)
        
        # contains unique vocabulary
        vocabulary=  list(set(cleaned_words))
        
        # lemmatize each word from each tokenzied sentences 
        tokens_words_sentences_lem = [lemmatize_preprocess(text = token_sentence) for token_sentence in tokens_sentences]
            
        

    return corpus_lem_words, cleaned_words, vocabulary, tokens_sentences, tokens_words_sentences, tokens_words_sentences_lem

# check = text_cleaning(txt = example_txt, stop_words= stop_words )



# after text cleaning and tokenized, implemnt words frequency analysis and TD-IDF analysis
# calculate the frequencies of each words from the whole text
# link: https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
# takethe top 30% of the keywords
# https://www.andyfitzgeraldconsulting.com/writing/keyword-extraction-nlp/
# https://kavita-ganesan.com/python-keyword-extraction/#.YhLokO7MIeZ

# returns words frequency in a whole text documents 
def TF_whole_text(word_token):
    
    str2 = []
    # loop till string values present in list str
    for i in word_token:             
        # checking for the duplicacy
        if i not in str2:
            # insert value in str2
            str2.append(i) 
    wrd_freq_cnt = []
    keywords = []
    for i in range(0, len(str2)):
        keywords.append(str2[i])
        wrd_freq_cnt.append(word_token.count(str2[i]))
    
    text_total_len = len(keywords)
    words_freq = pd.DataFrame()
    words_freq["vocab"] = keywords
    words_freq["word_doc_frequencies_cnt"] = wrd_freq_cnt
    words_freq["word_doc_frequency_pct"] = np.round(words_freq["word_doc_frequencies_cnt"]/text_total_len, 4)
    
    words_freq = words_freq.sort_values(by = "word_doc_frequency_pct", axis= 0, ascending = False)
    
    words_freq = words_freq.reset_index()
    
    
#     words_freq.head( int(text_total_len * 0.3)) 
    
    return  words_freq.iloc[: , 1:]


# after text cleaning and tokenized, implemnt calcuation on Term Frequency and Inverse document frequency
# extract a list list of top TF-IDF terms 
def TF_IDF_text(vocab2, sentences):

    Tf_idf = Pipeline([('count', CountVectorizer(vocabulary=vocab2)),
                  ('tfid', TfidfTransformer(smooth_idf=True,use_idf=True))]).fit(sentences)
    
    df_tf_idf = pd.DataFrame()

    df_tf_idf["vocab"] = vocab2
    df_tf_idf["tf_idf"] = list(Tf_idf['tfid'].idf_)
    
    df_tf_idf = df_tf_idf.sort_values(by='tf_idf', axis = 0)
    
    df_tf_idf=df_tf_idf.reset_index()
    
    return df_tf_idf.iloc[: , 1:]


# https://www.onely.com/blog/what-is-tf-idf/
def TF_IDF_text_doc(word_token, vocab, sentences ):
    
    part1 = TF_whole_text(word_token = word_token)
#     print(part1.head(10))
    part2 = TF_IDF_text(vocab2 = vocab, sentences = sentences)
    
    res = part2.merge(part1, how ='left', on='vocab' ).sort_values(by='tf_idf' , axis = 0)
    
    return res

# apply the functions
# df_temp = TF_IDF_text_doc(word_token = check[1], vocab = check[2],sentences = check[3] )
# df_temp


# based on the lemmatized token, word tokens and sentecnes tokens ==> create a bag of words, then build a topics model to extract topics in each text 
# https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
# The weights reflect how important a keyword is to that topic.
# extract topics with LDA
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def topic_extract(lemmatized_words_sentences, text_source = 'unknown'):
    dictionary = gensim.corpora.Dictionary(lemmatized_words_sentences)

    corpus = [dictionary.doc2bow(doc) for doc in lemmatized_words_sentences]

    

    lda_model =  gensim.models.ldamodel.LdaModel(corpus, 
                                       num_topics = 10, 
                                       id2word = dictionary,                                    
                                       passes = 10)
    #     doc_lda = lda_model[corpus]
    # return the top 10 topics
    
    res = pd.DataFrame()
    
    topics = lda_model.print_topics()
    res["source_identifier"] = [text_source] * len(topics)
    res['topics_extrcated'] = topics
    return res


# topic_extract(lemmatized_words_sentences = check[-1],  text_source = 'birds_intro_peafowl_t')



# extract the sentiment from text and sentences
# text blob: https://www.pluralsight.com/guides/natural-language-processing-extracting-sentiment-from-text-data
# https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524#:~:text=TextBlob%20returns%20polarity%20and%20subjectivity,1%20defines%20a%20positive%20sentiment.&text=Subjectivity%20lies%20between%20%5B0%2C1,information%20contained%20in%20the%20text.
# pip install textblob
# Polarity: Polarity lies between [-1,1], -1 defines a negative sentiment 
# and 1 defines a positive sentiment.Negation words reverse the polarity. 
# TextBlob has semantic labels that help with fine-grained analysis. 
#  For example — emoticons, exclamation mark, emojis, etc

# objectivity: Subjectivity lies between [0,1]. Subjectivity quantifies the amount of personal opinion and factual information contained in the text.
# The higher subjectivity means that the text contains personal opinion rather than factual information.

# TextBlob has one more parameter — intensity. TextBlob calculates subjectivity by looking at the ‘intensity’. Intensity determines if a word modifies the next word. For English, adverbs are used as modifiers (‘very good’).

def sentiment_label(sentiment_score):
    if sentiment_score < 0:
        return "Negative"
    elif sentiment_score == 0:
        return "Neutral"
    else:
        return "Positive"
def subject_label(subjectivity):
    if subjectivity > 0 and subjectivity < 0.5:
        return "More Factual"
    elif subjectivity == 0:
        return "Factual"
    elif subjectivity == 1:
        return "Opinion"
    else:
        return "More Opinion"
    
def extract_sentiment_score(text):

    text = TextBlob(text)
    polairty = text.sentiment[0]
    subjectivity = text.sentiment[1]
    
    return subjectivity, sentiment_label(polairty),  polairty,  subject_label(subjectivity) 

# extract_sentiment_score(text = example_txt)
 