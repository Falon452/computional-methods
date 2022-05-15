import os
import re
import time
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import diags
from sklearn.preprocessing import normalize
from scipy.sparse import save_npz
from scipy.sparse.linalg import svds
from nltk.stem import PorterStemmer


start = time.time()
porter_stemmer = PorterStemmer()

dataset_path = "./documents/"
NO_FILENAMES = 9300



def save_vocabulary(path, dict):
    with open(path, "w", encoding="utf8") as f:
        for word, ix in dict.items():
            f.write(word + "," + str(ix) + "\n")


def save_filenames(path, names):
    with open(path, "w", encoding="utf8") as f:
        for filename in names:
            f.write(filename + "\n")


def process_line(line):
    line = re.sub(r"[^a-zA-Z]", " ", line.lower()).split()
    stemmed_words = [porter_stemmer.stem(word) for word in line]
    return stemmed_words


def angle_between_vectors(v: np.array, u: np.array):
    dot_product = v @ u.T
    lengths_multiplied = np.sqrt(v @ v.T) * np.sqrt(u @ u.T)
    res = dot_product / lengths_multiplied
    return res


def get_dataset(filenames):
    dataset = []
    for filename in filenames:
        path = dataset_path + filename
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()[1:]
            dataset.append(" ".join(lines))
    return dataset


if __name__ == "__main__":
    print('getting dataset')

    filenames = os.listdir(dataset_path)[:NO_FILENAMES]

    dataset = get_dataset(filenames)

    count_vect = CountVectorizer(
        max_df=0.75,  # mark as stop word if word is in 75% docs
        tokenizer=process_line,
        max_features=8000
    )

    print("fit&transform count_vect")
    M = count_vect.fit_transform(dataset)

    print("tfidf")
    tfidf_transformer = TfidfTransformer()
    M = tfidf_transformer.fit_transform(M)

    print(M.shape)

    print("normalizing")
    normalize(M, axis=1, norm='l2', copy=False)  # redundant - dataset is already normalized

    print("saving matrix")
    save_npz("M.npz", M)

    print("saving vocabulary")
    save_vocabulary("vocabulary.txt", count_vect.vocabulary_)

    print("saving filenames")
    save_filenames("filenames.txt", filenames)

    print("singular values")
    NO_SINGULAR_VALUES = 10
    for i in [2**i for i in range(2, 12)]:
        u, s, vT = svds(M, k=i)
        np.save(f"u{i}", u)
        np.save(f"s{i}", s)
        np.save(f"vT{i}", vT)

    end = time.time()
    print(end - start)