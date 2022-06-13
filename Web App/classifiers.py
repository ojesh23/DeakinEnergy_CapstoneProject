"""
This python file contains all the text classifier models
"""
################## Imports ##################
import pickle
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_text
from tensorflow.keras.models import model_from_json
from keras_lookahead import Lookahead
from tensorflow.keras.preprocessing import sequence

classes = ['AF Other', 'Animal', 'Conductor', 'Connection', 'Crossarm', 'Dug up', 'Fuse', 'Installation', 'Lightning',
           'OH Cable', 'Other', 'Pole', 'Trees', 'UG Cable', 'Vehicle']

################## Random Forest ##################
class RandomForest:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.RF_MODEL_PATH = 'Models/rf/random_forest.pickle'
        self.TFIDF_ENCODER_PATH = 'Models/rf/TFIDF_1000.pickle'

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
        classes = rf_model.classes_
        return category, pred_proba, (classes, probability)


################## Stochastic Gradient Descent ##################
class SGD:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.SGD_MODEL_PATH = 'Models/sgd/sgd.pickle'
        self.TFIDF_ENCODER_PATH = 'Models/sgd/TFIDF_10000.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: SGD model and TFIDF encoder
        """
        # load model and encoder
        sgd_model = pickle.load(open(self.SGD_MODEL_PATH, 'rb'))
        tfidf_1000 = pickle.load(open(self.TFIDF_ENCODER_PATH, 'rb'))
        return sgd_model, tfidf_1000

    def sgd_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        sgd_model, tfidf_1000 = self.unpack_files()
        tfidf = tfidf_1000.transform(clean_text)
        category = sgd_model.predict(tfidf)[0]
        probability = sgd_model.predict_proba(tfidf)
        pred_proba = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, pred_proba, probability


################## Support Vector Machine ##################
class SVC:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.SVC_MODEL_PATH = 'Models/svc/svc.pickle'
        self.TFIDF_ENCODER_PATH = 'Models/svc/tf_5000.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: SVC model and TFIDF encoder
        """
        # load model and encoder
        svc_model = pickle.load(open(self.SVC_MODEL_PATH, 'rb'))
        tf_5000 = pickle.load(open(self.TFIDF_ENCODER_PATH, 'rb'))
        return svc_model, tf_5000

    def svc_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        svc_model, tf_5000 = self.unpack_files()
        tfidf = tf_5000.transform(clean_text)
        category = svc_model.predict(tfidf)[0]
        probability = svc_model.predict_proba(tfidf)
        pred_proba = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, pred_proba, probability


################## Recurrent Neural Network ##################


class RNN:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.RNN_MODEL_PATH = 'Models/rnn/model.h5'
        self.JSON_FILE_PATH = 'Models/rnn/model.json'
        self.TOKENIZER_FILE_PATH = 'Models/rnn/tok.pickle'
        self.LABEL_ENCODER_FILE_PATH = 'Models/rnn/le.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: rnn model, weights, tokenizer and label encoder
        """
        # load model and encoder

        json_file = open(self.JSON_FILE_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.RNN_MODEL_PATH)
        # load the tokenizer
        tok = pickle.load(open(self.TOKENIZER_FILE_PATH, "rb"))
        # load the label encoder
        le = pickle.load(open(self.LABEL_ENCODER_FILE_PATH, "rb"))
        return le, tok, loaded_model

    def rnn_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        le, tok, loaded_model = self.unpack_files()
        optimiser = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6,
                                             amsgrad=False)
        lk = Lookahead(optimiser, sync_period=5, slow_step=0.5, name='Lookahead')
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=lk.optimizer, metrics=['accuracy'])

        sequences = tok.texts_to_sequences(clean_text)
        sequence_matrix = sequence.pad_sequences(sequences, maxlen=50)
        y_pred = loaded_model.predict_classes(sequence_matrix)
        category = le.inverse_transform(y_pred)[0]
        probability = loaded_model.predict_proba(sequence_matrix)
        rnn_prob = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, rnn_prob, probability


################## Logistic Regression ##################

class LOGISTICREGRESSION:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.LR_MODEL_PATH = 'Models/lr/LR.pickle'
        self.TFIDF_ENCODER_PATH = 'Models/lr/TFIDF_500.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: random forest model and TFIDF encoder
        """
        # load model and encoder
        lr_model = pickle.load(open(self.LR_MODEL_PATH, 'rb'))
        vect = pickle.load(open(self.TFIDF_ENCODER_PATH, 'rb'))
        return lr_model, vect

    def lr_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        lr_model, vect = self.unpack_files()
        vect = vect.transform(clean_text)
        category = classes[lr_model.predict(vect)[0]]
        probability = lr_model.predict_proba(vect)
        pred_proba = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, pred_proba, probability

################## Logistic Regression ##################

class XgBoost:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.XGB_MODEL_PATH = 'Models/xgb/XGB.pickle'
        self.VECT_ENCODER_PATH = 'Models/xgb/TFIDF_500.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: random forest model and TFIDF encoder
        """
        # load model and encoder
        xgb_model = pickle.load(open(self.XGB_MODEL_PATH, 'rb'))
        countvect = pickle.load(open(self.VECT_ENCODER_PATH, 'rb'))
        return xgb_model, countvect

    def xgb_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        xgb_model, vect = self.unpack_files()
        vect = vect.transform(clean_text)
        category = classes[xgb_model.predict(vect)[0]]
        probability = xgb_model.predict_proba(vect)
        pred_proba = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, pred_proba, probability


################## Voting Classifier ##################

class VotingClassifier:
    def __init__(self, text):
        """
        args: text, text from user input fields
        """
        self.text = text
        self.VOTING_MODEL_PATH = 'Models/voting/voting.pickle'
        self.COUNTVECT_ENCODER_PATH = 'Models/voting/TFIDF_10000.pickle'

    def unpack_files(self):
        """
        load pickle files
        returns: random forest model and TFIDF encoder
        """
        # load model and encoder
        voting_model = pickle.load(open(self.VOTING_MODEL_PATH, 'rb'))
        countvect = pickle.load(open(self.COUNTVECT_ENCODER_PATH, 'rb'))
        return voting_model, countvect

    def voting_prediction(self):
        """
        return: incident category and maximum prediction probability
        """
        clean_text = preprocess_text(self.text)
        voting_model, countvect = self.unpack_files()
        countvect = countvect.transform(clean_text)
        category = classes[voting_model.predict(countvect)[0]]
        probability =voting_model.predict_proba(countvect)
        pred_proba = str(np.round(np.max(probability) * 100, 2)) + "%"
        return category, pred_proba, probability
