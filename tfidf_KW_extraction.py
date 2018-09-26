from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
from itertools import islice
import nltk
import numpy as np

class Tfidf_Pos_keywords:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(corpus)
        self.good_pos = ['NN', 'JJ', 'NNS', 'NNP', 'VBG', 'VBP', \
            'VBD', 'VBZ', 'VBN', 'IN', 'RB', 'CD']
        
    def __call__(self, *args, **kwargs):
        return self.keywords(*args, **kwargs)

    def _tfidfed_keywords(self, document):
        """
        finds words and their corresponding tfidf scores in `document`.
        """
        keywords = dict()
        transformed_doc = self.vectorizer.transform([document]).toarray()[0]
        for w in document.split():
            try:
                word_id = self.vectorizer.vocabulary_[w]
                keywords[w] = transformed_doc[word_id]
            except KeyError:
                pass
        return keywords

    def _pos_tags(self, text):
        text = nltk.word_tokenize(text)
        return nltk.pos_tag(text)
    
    def _is_good_pos(self, tag):
        return tag in self.good_pos
    
    def _candidate_keywords(self, text):
        words = []
        # filter promising words
        for w, t in self._pos_tags(text):
            if self._is_good_pos(t):
                words.append(w)
            else:
                words.append("")

        candidates = []
        candidate = []
        # assemble tokens to candidates
        for word in words:
            if word != "":
                candidate.append(word)
            else:
                if len(candidate) > 0:
                    candidates.append(candidate)
                    candidate = []
        else: # appending the last element
            if len(candidate) > 0:
                candidates.append(candidate)
        return candidates
    
    def _score_candidate(self, word_scores, candidate):
        return np.array([word_scores[t] if t in word_scores else 0 for t in candidate]).mean()
    
    def _score_candidates(self, word_scores, candidates):
        scored = set([(" ".join(c), self._score_candidate(word_scores, c)) for c in candidates])
        ordered = sorted(scored, key=lambda x: x[1], reverse=True)
        return ordered
    
    def keywords(self, text, num_kwds=None, scores=False):
        """
        extracts keywords based on their tfidf score. ie the words 
        with the highest tfidf score are selected. 
        """        
        text = text.lower()
        word_scores = self._tfidfed_keywords(text)
        candidates = self._candidate_keywords(text)
        sorted_kwds = self._score_candidates(word_scores, candidates)

        if scores == False:
            sorted_kwds = [kw for kw, scores in sorted_kwds]

        return list(islice(sorted_kwds, num_kwds))