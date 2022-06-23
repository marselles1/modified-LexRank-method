import lexrank
from lexrank import LexRank
from path import Path
import numpy as reverse
import numpy as np
import os
import fitz

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import re
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords


from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import punkt
from itertools import combinations
from nltk import ngrams
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import io
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def pdf_to_text(
    input_pdf,
    output_txt
  ):
  i_f = open(input_pdf,'rb')
  resMgr = PDFResourceManager()
  retData = io.StringIO()
  TxtConverter = TextConverter(resMgr,retData, laparams= LAParams())
  interpreter = PDFPageInterpreter(resMgr,TxtConverter)
  for page in PDFPage.get_pages(i_f):
      interpreter.process_page(page) 
  txt = retData.getvalue()
  print(txt)
  with open(output_txt,'w', encoding = 'UTF-8') as of:
        of.write(txt)

def delete_reference_intro(
    text
  ):
  half_cleared_text = re.sub(r'(.*INTRODUCTION)|(.*Introduction)','', text)
  half_cleared_text_1 = re.sub(r'\((cid:..)\)|\((cid:.)\)','', half_cleared_text)
  cleared_text = re.sub(r'(References.*)|(REFERENCES.*)','', half_cleared_text_1)
  return cleared_text

def delete_stopwords(
    sents,
    stop_words
  ): 
  for i,_ in enumerate(sents):
      sents[i] = sents[i].split()
      for word in list(sents[i]):
          if word.lower() in stop_words or not word.isalnum():
              sents[i].remove(word)
  for i,_ in enumerate(sents):
    sents[i] = ' '.join(sents[i])
  return sents

def calculate_bigramms(
    sents_dict,
    sents
  ):
  bigramms_freq = {}
  for key in sents_dict.keys():
    try: 
      bigramms = list(ngrams(sents[key].split(),2))
    except Exception:
      continue
    sents_dict[key] = bigramms
    for bigramm in bigramms:
      if bigramm in bigramms_freq.keys():
        bigramms_freq[bigramm] += 1
      else:
        bigramms_freq[bigramm] = 1
  return bigramms_freq

def search_top_sents(
    top_bigramms,
    top_sents,
    sents
  ):
  for bigramm in top_bigramms:
    bigramm =  ' '.join(list(bigramm))
    for i,sent in enumerate(sents):
      if bigramm in sent: 
        if sent in top_sents.keys():
          top_sents[sent] += 1
        else:
          top_sents[sent] = 1
  return top_sents

def methods_lexrank(
    lxr, 
    sents_lexrank, 
    calc_res, 
    lexrank_ids
  ):
  summary = lxr.get_summary(sents_lexrank, summary_size=calc_res, threshold=.1)
  for g in summary:
    for i,sent in enumerate(sents_lexrank):
      if g == sent:
        lexrank_ids.append(i)
  return lexrank_ids

def calculate_top_sents(
    sents, 
    top_sents, 
    lexrank_ids, 
    top_sents_ids, 
    sents_num, 
    calc_res
  ):
  for i in range(len(sents)):
    for sentence in list(top_sents.keys()):
      if sentence == sents[i]:
        print(i)
        if i not in lexrank_ids:
          top_sents_ids.append(i)
      if len(top_sents_ids) == sents_num-calc_res:
        print(top_sents_ids)
        return top_sents_ids

def main():
  input_pdf = '/Users/gmars/Desktop/data/Shelekhov.pdf'
  output_txt = '/Users/gmars/Desktop/data/NewFolder/sample.txt'

  output_txt = pdf_to_text(input_pdf,output_txt)

  with open('/Users/gmars/Desktop/data/NewFolder/sample.txt','r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\n',' ')

  print('Введите нужное количество предложений для аннотации:')
  sents_num = int(input())  
  
  calc_res = sents_num//2

  cleared_text = delete_reference_intro(text)

  stop_words = set(stopwords.words('english'))
  sents = tokenize.sent_tokenize(cleared_text) 

  sents = delete_stopwords(sents, stop_words)

  sents_dict =  {i : [] for i in range(len(sents))}

  bigramms_freq = calculate_bigramms(sents_dict, sents)

  bigramms_freq = dict(sorted(bigramms_freq.items(), key=lambda item: item[1],reverse=True))
  top_bigramms = list(bigramms_freq.keys())[:10]

  top_sents_ids = []
  top_sents = {}

  top_sents = search_top_sents(top_bigramms, top_sents, sents)

  top_sents = dict(sorted(top_sents.items(), key=lambda item: item[1],reverse=True))

  lxr = LexRank(text, stopwords=stop_words)

  orig_text = tokenize.sent_tokenize(cleared_text)
  sents_lexrank = tokenize.sent_tokenize(cleared_text)
  lexrank_ids = []

  lexrank_ids = methods_lexrank(lxr, sents_lexrank, calc_res, lexrank_ids)

  top_sents_ids = calculate_top_sents(sents, top_sents, lexrank_ids, top_sents_ids, sents_num, calc_res)

  annotation_ids = lexrank_ids + top_sents_ids

  with open('/Users/gmars/Desktop/data/NewFolder/annotation.txt', 'w', encoding='utf-8') as f:
    for ids in annotation_ids:
      f.write(orig_text[ids]+'\n')
      print(orig_text[ids]+'\n')

if __name__ == '__main__':
  main()