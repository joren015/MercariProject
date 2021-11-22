import gc
import pickle
from os import makedirs

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

pd.set_option('max_colwidth', 200)


# Create split_cat() function that returns cateogires (dae, jung, so) called by "apply lambda"
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null', 'Other_Null', 'Other_Null']


def fit_and_save_vectorizer(X, vectorizer, folder_name):
    makedirs(folder_name, exist_ok=True)
    X_vectorized = vectorizer.fit_transform(X)
    with open("{}/vectorizer.pkl".format(folder_name), 'wb') as f:
        pickle.dump(vectorizer, f)

    with open("{}/X.pkl".format(folder_name), 'wb') as f:
        pickle.dump(X, f)

    with open("{}/X_vectorized.pkl".format(folder_name), 'wb') as f:
        pickle.dump(X_vectorized, f)

    return vectorizer, X_vectorized


def preprocess_light_gbm(nrows=-1):
    if nrows > 0:
        mercari_df = pd.read_csv('./data/train.tsv', sep='\t', nrows=nrows)
    else:
        mercari_df = pd.read_csv('./data/train.tsv', sep='\t')

    # Calls split_cat() function above and create cat_dae, cat_jung, cat_so columns in mercari_df
    mercari_df['category_list'] = mercari_df['category_name'].apply(
        lambda x: split_cat(x))
    mercari_df['category_list'].head()

    mercari_df['cat_dae'] = mercari_df['category_list'].apply(lambda x: x[0])
    mercari_df['cat_jung'] = mercari_df['category_list'].apply(lambda x: x[1])
    mercari_df['cat_so'] = mercari_df['category_list'].apply(lambda x: x[2])

    mercari_df.drop('category_list', axis=1, inplace=True)

    # Handling Null Values
    mercari_df['brand_name'] = mercari_df['brand_name'].fillna(
        value='Other_Null')
    mercari_df['category_name'] = mercari_df['category_name'].fillna(
        value='Other_Null')
    mercari_df['item_description'] = mercari_df['item_description'].fillna(
        value='Other_Null')

    gc.collect()

    print("Vectorizing name")
    # Convert "name" with feature vectorization
    fit_and_save_vectorizer(mercari_df["name"],
                            CountVectorizer(max_features=30000),
                            "data/name_count_vectorizer")

    print("Vectorizing item_description")
    # Convert "item_description" with feature vectorization
    fit_and_save_vectorizer(
        mercari_df['item_description'],
        TfidfVectorizer(max_features=50000,
                        ngram_range=(1, 3),
                        stop_words='english'),
        "data/item_description_tfidf_vectorizer")

    # Convert each feature (brand_name, item_condition_id, shipping) to one-hot-encoded sparse matrix
    # Convert each feature (cat_dae, cat_jung, cat_so) to one-hot-encoded spare matrix
    for col in [
            "brand_name", "item_condition_id", "shipping", "cat_dae",
            "cat_jung", "cat_so"
    ]:
        print("Vectorizing {}".format(col))
        fit_and_save_vectorizer(mercari_df[col],
                                LabelBinarizer(sparse_output=True),
                                "data/{}_label_binarizer".format(col))