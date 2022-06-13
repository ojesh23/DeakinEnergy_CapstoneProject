"""
This python file contains all the text classifier models
"""
################## Imports ##################
import pickle
import numpy as np
from preprocessing import preprocess_text

################## Random Forest ##################
class RandomForest:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.RF_MODEL_PATH = 'Models/T3/RF/random_forest.pickle'
        self.TFIDF_ENCODER_PATH = 'Models/T3/RF/TFIDF_1000.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: random forest model and TFIDF encoder
        """
        # load model and encoder
        rf_model = pickle.load(open(self.RF_MODEL_PATH, 'rb'))
        tfidf_1000 = pickle.load(open(self.TFIDF_ENCODER_PATH, 'rb'))
        return rf_model, tfidf_1000

    def rf_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        rf_model, tfidf_1000 = self.unpack_files()
        tfidf = tfidf_1000.transform(clean_text)
        category = rf_model.predict(tfidf)[0]
        probability = rf_model.predict_proba(tfidf)
        pred_proba = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, pred_proba



