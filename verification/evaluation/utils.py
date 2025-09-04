import re
import string
import nltk
import numpy as np
from nltk.tokenize import word_tokenize


def setup_nltk_arabic():
    """Download all required NLTK resources for Arabic"""
    resources = [
        "punkt",
        "punkt_tab",  # This is the missing one!
        "averaged_perceptron_tagger",
        "wordnet",
        "omw-1.4",  # For multilingual wordnet
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")


def arabic_tokenize(text):
    """Custom Arabic tokenizer that handles Arabic-specific requirements"""
    # Remove diacritics (tashkeel)
    arabic_diacritics = re.compile(r"[\u064B-\u0652\u0670\u0640]")
    text = arabic_diacritics.sub("", text)

    # Remove punctuation
    arabic_punctuation = "؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؜؝؞؟؀"
    all_punctuation = string.punctuation + arabic_punctuation

    for p in all_punctuation:
        text = text.replace(p, " ")

    # Split and clean
    tokens = [token.strip() for token in text.split() if token.strip()]

    # Filter Arabic/English only
    cleaned_tokens = []
    for token in tokens:
        if re.search(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z]",
            token,
        ):
            cleaned_tokens.append(token)

    return cleaned_tokens


def safe_arabic_tokenize(text):
    """
    Safe Arabic tokenizer with fallbacks
    """
    # Try NLTK Arabic tokenizer first
    try:
        tokens = word_tokenize(text, language="arabic")
        return tokens
    except LookupError:
        return arabic_tokenize(text)
    except Exception as e:
        print(f"NLTK tokenization failed ({e}), using custom tokenizer...")
        return arabic_tokenize(text)


def pairwise_meteor_arabic(candidate, reference):
    """
    METEOR score for Arabic with proper tokenization and fallbacks
    """
    try:
        candidate_tokens = safe_arabic_tokenize(candidate)
        reference_tokens = safe_arabic_tokenize(reference)

        if not candidate_tokens or not reference_tokens:
            return 0.0

        return nltk.translate.meteor_score.single_meteor_score(
            reference_tokens, candidate_tokens
        )
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def pairwise_meteor(candidate, reference):
    return pairwise_meteor_arabic(candidate, reference)


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    scores = np.empty((len(src_data), len(tgt_data)))

    for i, src in enumerate(src_data):
        for j, tgt in enumerate(tgt_data):
            scores[i][j] = metric(src, tgt)

    return scores
