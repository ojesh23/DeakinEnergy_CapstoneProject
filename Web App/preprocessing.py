"""

This python file will contain only text pre-processing pipelines

"""
################## Imports ##################
import texthero as hero
from texthero import preprocessing
import pandas as pd
import nltk
nltk.data.path.append("nltk data")

################## Text Cleaning and Pre-processing (TextHero) ##################
def preprocess_text(text):
    """
    Curently in use for following classifiers:
    - Random Forest, SGD, SVC, RNN, LR
    """
    # cleaning steps
    cleaning_pipeline = [preprocessing.fillna, preprocessing.lowercase, preprocessing.remove_whitespace,
                         preprocessing.remove_punctuation, preprocessing.remove_urls, preprocessing.remove_brackets,
                         preprocessing.remove_stopwords, preprocessing.remove_digits,
                         preprocessing.remove_angle_brackets, preprocessing.remove_curly_brackets, preprocessing.stem]

    # apply pipeline to text
    text = pd.Series(text)
    clean_text = text.pipe(hero.clean, cleaning_pipeline)

    return clean_text
