import contextlib
import re
import string

import nltk
import numpy as np
from nltk.tokenize import word_tokenize

_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_ARABIC_PUNCT = "؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؜؝؞؟؀"
_ARABIC_CHARS_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿a-zA-Z]")


def setup_nltk_arabic() -> None:
    for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]:
        with contextlib.suppress(Exception):
            nltk.download(resource, quiet=True)


def arabic_tokenize(text: str) -> list[str]:
    text = _ARABIC_DIACRITICS.sub("", text)
    for p in string.punctuation + _ARABIC_PUNCT:
        text = text.replace(p, " ")
    return [t for t in text.split() if t.strip() and _ARABIC_CHARS_RE.search(t)]


def safe_arabic_tokenize(text: str) -> list[str]:
    try:
        tokens: list[str] = list(word_tokenize(text, language="arabic"))
        return tokens
    except LookupError:
        return arabic_tokenize(text)
    except Exception:
        return arabic_tokenize(text)


def pairwise_meteor(candidate: str, reference: str) -> float:
    try:
        c_tokens = safe_arabic_tokenize(candidate)
        r_tokens = safe_arabic_tokenize(reference)
        if not c_tokens or not r_tokens:
            return 0.0
        return float(nltk.translate.meteor_score.single_meteor_score(r_tokens, c_tokens))
    except Exception:
        return 0.0


def compute_all_pairwise_scores(
    src_data: list[str], tgt_data: list[str], metric: object
) -> np.ndarray:
    scores = np.empty((len(src_data), len(tgt_data)))
    for i, src in enumerate(src_data):
        for j, tgt in enumerate(tgt_data):
            scores[i][j] = metric(src, tgt)  # type: ignore[operator]
    return scores
