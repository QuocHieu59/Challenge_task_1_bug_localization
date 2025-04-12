import csv
import re
import os
import random
import timeit
import string
import numpy as np
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def tsv2dict(tsv_path):
    """ Converts a tab separated values (tsv) file into a list of dictionaries

    Arguments:
        tsv_path {string} -- path of the tsv file
    """
    reader = csv.DictReader(open(tsv_path, "r"), delimiter="\t")
    dict_list = []
    for line in reader:
        line["files"] = [
            os.path.normpath(f)
            for f in line["files"].strip().split()
            if f.endswith(".java")
        ]
        line["raw_text"] = line["summary"] + line["description"]
        # line["summary"] = clean_and_split(line["summary"][11:])
        # line["description"] = clean_and_split(line["description"])
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S"
        )

        dict_list.append(line)
    return dict_list

bug_reports = tsv2dict("D:/Me-hi/20242/Phan_mem_use_LLM/test/bug-localization-by-dnn-and-rvsm/Data_bug/SWT.txt")

print(bug_reports[0])
for i, br in enumerate(bug_reports):
    print(br)