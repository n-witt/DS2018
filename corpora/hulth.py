import pandas as pd
import os
import re
from os.path import join

CORPUS_DIR = 'corpora/hulth2003'

def load():
    df = pd.DataFrame(columns=['abstr', 'contr', 'uncontr'])
    for base_dir, dirs, files in os.walk(CORPUS_DIR):
        for file in files:
            with open(join(base_dir, file)) as fh:
                idx, kind = file.split('.')
                text = fh.read()
                text = re.sub(r"\s+", " ", text, flags = re.UNICODE) # remove whitespaces
                df.loc[int(idx), kind] = text

    df["uncontr"] = df["uncontr"].apply(str.strip)
    df["contr"] = df["contr"].apply(str.strip)
    df.rename(
        columns={"abstr": "abstract", "uncontr": "keywords"},
        inplace=True)
    return df
