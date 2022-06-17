###################################
# grammer/spelling correct
###################################

import pandas as pd
import numpy as np
# from NLP_Data_Engineer_Functions import text_cleaning, stop_words
import nltk
from nltk.tokenize import sent_tokenize


# perform edit distance in python
#levemstain distance used to measure the infoamtion difference
# https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/

# String Similarity Measures
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
#     print (matrix)
    return (matrix[size_x - 1, size_y - 1])
# seq1 = 'I walk to the store and I bought milk'
# seq2 =  'I was walking to the store and I bought milk.'

# levenshtein(seq1, seq2)

# return different words from 2 sentences
def UncommonWords(A, B):
  
    # count will contain all the word counts
    count = {}
      
    # insert words of string A to hash
    for word in A.split():
        count[word] = count.get(word, 0) + 1
      
    # insert words of string B to hash
    for word in B.split():
        count[word] = count.get(word, 0) + 1
  
    # return required list of words
    return [word for word in count if count[word] == 1]

# n-gram similarity measures: n-gram of a string s in any substring of s of length n. A simple measure would be to choose n and count the number of common n-grams between two strings s and t.
# https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/

# word counts

def word_cnt_compare(sentence1, sentence2):
    words1 = nltk.word_tokenize(sentence1) 
    words2 = nltk.word_tokenize(sentence2) 
    return len(words1) - len(words2)

# Implement Text Correction Technqiue in the function
# Word might me wrong
# Has one extra letter
# Missing one letter
# Has a single transposition i.e. a pair of adjacent letters in the word interchanged.
# symspellpy: https://github.com/mammothb/symspellpy
# grammerly API
# apply Gramformer to apply sentences corrections to each sentences 

# import spacy
# from gramformer import Gramformer
# import torch
# import en_core_web_sm
# from nltk.tokenize import word_tokenize
# nlp = spacy.blank("en")
# nlp = en_core_web_sm.load()

# choose to use the original raw sentences vs corrected sentences
def version_select(txt1, txt2, count, ratio):
    if count <= 4:
        return txt1
    else:
        if ratio <= 0.1:
            return txt2
        else:
            return txt1
    

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# implement Gramformer API to correct text sentences 
def Correct_Sentences(file_identifier, temp_txt):
    # set_seed(1212)
    gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector
    
    sentences = sent_tokenize(temp_txt)
    
    raw_sentences =  sentences 
    corrected_sentences = []
    Levenshtain_scores = []
    Jaccard_scores = []
    words_diff = []
    for i in range(0, len(raw_sentences)):
        raw_sentence = raw_sentences[i]
        corrected_sentence = list(gf.correct(raw_sentence, max_candidates=1))[0]
        corrected_sentences.append(corrected_sentence )
        Levenshtain_scores.append(levenshtein(seq1 = raw_sentence , seq2 = corrected_sentence))
        Jaccard_scores.append(nltk.jaccard_distance(set(raw_sentence), set(corrected_sentence) ))
        words_diff.append(word_cnt_compare(raw_sentence, corrected_sentence))
        
    # formulatae the corrected sentences back to txt chunk
    start_txt = ''
    for indx, sen in enumerate(corrected_sentences):
        if indx == 0:
            corrected_txt = start_txt + sen 
        else:
            corrected_txt = corrected_txt + " " + sen
    
    # form dataframe from all the outputs
    res_df = pd.DataFrame()
    res_df["file_name"] = [file_identifier] * len(corrected_sentences)
    res_df["raw_txt"] = [temp_txt] * len(corrected_sentences)
    res_df["raw_sentences"] = raw_sentences
    res_df["corrected_sentences"] = corrected_sentences
    res_df["corrected_txt"] = corrected_txt
    res_df["sentences_levenshtain_scores"] = Levenshtain_scores
    res_df["sentences_jaccard_scores"] = Jaccard_scores
    res_df["sentences_num_words_diff"] = words_diff
    res_df["sentences_num_words_diff_abs"] = np.abs(words_diff)
    
    raw_sentences_word_counts = [len(word_tokenize(i)) for i in list(res_df["raw_sentences"])]
    res_df["raw_sentences_word_counts"] = raw_sentences_word_counts
    res_df["correction_ratio"] = res_df["sentences_num_words_diff_abs"]/res_df["raw_sentences_word_counts"]
    

    # select to keep the corewcted txt vs keep the raw_txt
    res_df['final_corrected_version_sentences_txt'] = res_df.apply(lambda x: version_select(x.raw_sentences, x.corrected_sentences, x.correction_ratio, x.raw_sentences_word_counts ), axis=1)
    
    # formulatae the final corrected sentences back to txt chunk
    start_txt = ''
    for indx, sen in enumerate(list(res_df['final_corrected_version_sentences_txt'])):
        if indx == 0:
            final_txt = start_txt + sen 
        else:
            final_txt = corrected_txt + " " + sen
    
    res_df['final_corrected_version_sentences_txt'] = final_txt
    
    return res_df
    
    

# application 
# text_extracted_pd = pd.read_csv("Targeted_Output/Phase1/0_ready_for_scoring_batch1.csv")
# videos_files_lst = list(text_extracted_pd['video_file'])
# text_extracted_pd.head()

# result_corrected_df = pd.DataFrame()
# for file in files_identifiers:
    
#     temp_txt = text_extracted_pd[text_extracted_pd['video_file'] == file]['raw_txt'].values[0]
#     temp_pd = Correct_Sentences(file_identifier = file, temp_txt = temp_txt)
#     result_corrected_df = pd.concat([result_corrected_df, temp_pd ], axis = 0)


