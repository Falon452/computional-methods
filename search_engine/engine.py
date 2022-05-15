from pip import main
from scipy.sparse import load_npz
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import time
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS


porter_stemmer = PorterStemmer()

count_vect = CountVectorizer()

vocabulary = []
filenames = []


class QueryList(Resource):
    def get(self, student_id):
        start = time.time()
        print(student_id)
        res = query(student_id)
        end = time.time()
        print(end - start)
        return res


app = Flask(__name__)
api = Api(app)

CORS(app, resources={r"*": {"origins": "*"}})
api.add_resource(QueryList, '/<student_id>')


def query(text):
    query_count_vector = count_vect.fit_transform([text])
    return _get_k_closest_to_query(M, query_count_vector, filenames, 10)


def _process_line(line):
    line = re.sub(r"[^a-zA-Z]", " ", line.lower()).split()
    stemmed_words = [porter_stemmer.stem(word) for word in line]
    return stemmed_words


def _angle_between_vectors(v: np.array, u: np.array):
    dot_product = v @ u.T
    lengths_multiplied = np.sqrt(v @ v.T) * np.sqrt(u @ u.T)
    lengths_multiplied = lengths_multiplied.todense()
    res = dot_product / lengths_multiplied
    return res
    

def _index_in_descdening(arr, key):
    for i, x in enumerate(arr):
        if key > x:
            return i
    return len(arr)


def _get_k_closest_to_query(M, query_count_vector, filenames, k):
    closests_indices = []
    angles = []  # sorted descending
    no_docs = M.shape[0]

    for row_ix in range(no_docs):
        angle = _angle_between_vectors(M[row_ix], query_count_vector)[0][0]
        
        ix_to_insert = _index_in_descdening(angles, angle)

        if len(closests_indices) < k:
            angles.insert(ix_to_insert, angle)
            closests_indices.insert(ix_to_insert, row_ix)
        else:
            if ix_to_insert >= k:
                continue
            else:
                angles.insert(ix_to_insert, angle)
                closests_indices.insert(ix_to_insert, row_ix)
                if len(angles) > k:
                    angles.pop()
                    closests_indices.pop()

    links = []
    titles = []
    struct = []
    title = list(map(lambda index: filenames[index], closests_indices))
    title = list(map(lambda t: t.removesuffix(".txt"), title))
    title = list(map(lambda t: t.replace("_", " "), title))

    paths = list(map(lambda index: "./documents/" + filenames[index], closests_indices))
    for path in paths:
        with open(path, "r", encoding="utf8") as f:
            links.append(f.readline())

    for t, link in zip(title, links):
        struct.append({'title': t, 'url': link})

    return struct


def _load_filenames(path):
    with open(path, "r", encoding="utf8") as f:
        return list(map(lambda line: line.removesuffix("\n"), f.readlines()))


def _load_vocabulary(path):
    vocab = dict()
    with open(path, "r", encoding="utf8") as f:
        for line in f.readlines():
            word, ix = line.split(',')
            vocab[word] = int(ix)
    return vocab  


if __name__ == "__main__":
    filenames = _load_filenames("filenames.txt")
    vocabulary = _load_vocabulary("vocabulary.txt")

    M = load_npz("M.npz")
    # u5, s5, vT5 = np.load("u512.npy"), np.load("s512.npy"), np.load("vT512.npy")
    # M = u5 @ np.diag(s5) @ vT5

    # print(u5.shape, s5.shape, vT5.shape)


    count_vect = CountVectorizer(
        vocabulary=vocabulary,
        tokenizer=_process_line
    )

    
    app.run(debug=True)



    