# Got it from https://www.kaggle.com/neelshah18/scopusjournal/data
import pandas as pd

def load():
    df = pd.read_csv("corpora/scopus.csv")
    df.rename(
        columns={"Abstract": "abstract", "Author Keywords": "keywords"},
        inplace=True)
    return df[df["keywords"].notna()]
