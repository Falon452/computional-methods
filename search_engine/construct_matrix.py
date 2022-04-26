import os
import pandas as pd
import numpy as np
import collections
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
from sklearn.preprocessing import normalize

wordmap = dict()
wordset = []

doc_map = dict()
no_documents = 0
matrix = np.zeros(1)


def process_line(line):
    line = re.sub(r"[^a-zA-Z]", " ", line.lower()).split()
    line = list(filter(lambda word: 5 <= len(word) < 12, line))
    line = list(map(lambda word: ps.stem(word), line))
    return line


def save_wordset(wordset):
    with open('./saved/wordset.txt', 'w+') as file:
        file.write(' '.join(wordset))


def get_wordset():
    global wordset, no_documents
    wordset = np.union1d('', '')
    for filename in os.listdir('./documents'):
        no_documents += 1
        path = './documents/' + filename
        with open(path, 'r') as f:
            for line in f.readlines()[1:]:
                line = process_line(line)
                wordset = np.union1d(wordset, line)

    save_wordset(wordset)
    return wordset


def create_rows():
    global matrix

    for index, filename in enumerate(os.listdir('./documents')):
        doc_map[filename] = index
        path = './documents/' + filename

        row = np.zeros(shape=len(wordset))

        max_value_in_row = 1

        with open(path, 'r') as f:
            for line in f.readlines()[1:]:
                line = process_line(line)
                for word in line:
                    word_ix = wordmap.get(word, -1)
                    if (word_ix == -1):
                        continue

                    matrix[index][wordmap[word]] += 1
                    if matrix[index][wordmap[word]] > max_value_in_row:
                        max_value_in_row = matrix[index][wordmap[word]]


        matrix[index] = normalize(matrix[index][:, np.newaxis], axis=0, norm='l1').ravel()
        # matrix[index] = np.divide(matrix[index], max_value_in_row)
        # matrix[index] = matrix[index] /  max_value_in_row


def create_matrix():
    global matrix
    matrix = np.zeros(shape=(no_documents, len(wordset)), dtype=np.float64)
    print(matrix.shape)
    create_rows()

def main():
    get_wordset()
    create_wordmap()
    create_matrix()


def create_wordmap():
    global wordset, wordmap
    index = 0

    for word in wordset:
        wordmap[word] = index
        index += 1

if __name__ == '__main__':
    # index = 0
    #
    # wordset = get_wordset()
    # print(len(wordset))
    # for word in wordset:
    #     wordmap[word] = index
    #     index += 1
    # create_rows()
    # print(wordmap)
    main()
    for i in matrix[0]:
        if i != 0:
            print(i)
    # print(sum(matrix[0]))
    # print(sum(matrix[1]))
    # print(sum(matrix[2]))
    # print(sum(matrix[3]))
    # print(sum(matrix[4]))



