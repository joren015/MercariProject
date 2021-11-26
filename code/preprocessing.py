import pickle
from os import makedirs

import pandas as pd

pd.set_option('max_colwidth', 200)


# Create split_cat() function that returns cateogires (dae, jung, so) called by "apply lambda"
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null', 'Other_Null', 'Other_Null']


def fit_and_save_vectorizer(X, vectorizer, folder_name, non_sklearn=False):
    makedirs(folder_name, exist_ok=True)
    if non_sklearn:
        vectorizer.fit_on_texts(X)
        with open("{}/vectorizer.pkl".format(folder_name), 'wb') as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        vectorizer.fit(X)
        with open("{}/vectorizer.pkl".format(folder_name), 'wb') as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    return vectorizer


def load_and_apply_vectorizer(vectorizer_path, col):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return vectorizer.transform(col)
