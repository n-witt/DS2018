import pandas as pd
import os
import re
from os.path import join

CORPUS_DIR = 'corpora/kp20k/'

def load():
    df_train = pd.read_json(os.path.join(CORPUS_DIR, "kp20k_training.json"), lines=True)
    df_test = pd.read_json(os.path.join(CORPUS_DIR, "kp20k_testing.json"), lines=True)
    df_validate = pd.read_json(os.path.join(CORPUS_DIR, "kp20k_validation.json"), lines=True)

    df = df_train.append(df_test).append(df_validate)

    df.rename(columns={"keyword": "keywords"}, inplace=True)
    df["keywords"] = df["keywords"].apply(lambda x: x.replace(";", "; "))
    return df