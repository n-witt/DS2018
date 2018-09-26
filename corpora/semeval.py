import pandas as pd
import os
from os.path import join
import io

CORPUS_DIR = "corpora/semEval/data"

def convert_keywords(text):
    kwds = [kw.split("\t")[-1] for kw in text.split("\n") if "Process" not in kw]
    kwds = [kw for kw in kwds if kw != ""]
    return "; ".join(kwds)

def load():
    df = pd.DataFrame()
    for base_dir, dirs, files in os.walk(CORPUS_DIR):
        for file in files:
            with io.open(join(base_dir, file), encoding="utf8") as fh:
                idx, kind = file.split('.')
                text = fh.read()
                df.loc[idx, kind] = text
    df.rename(columns={"txt": "abstract", "ann": "keywords"}, inplace=True)
    df["keywords"] = df["keywords"].apply(convert_keywords)
    return df
