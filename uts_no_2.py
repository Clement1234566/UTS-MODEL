# -*- coding: utf-8 -*-
"""UTS No 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MrxRCroGhKtfykFi8GqsHNYrfLEmTDXs
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.x = self.df[self.df.columns.drop(['loan_status'])]
        self.y = self.df['loan_status']

    def handle_missing_values(self):
        self.x['person_income'].fillna(self.x['person_income'].median(), inplace=True)

    def handle_categorical_data(self):
        self.x['person_gender'] = self.x['person_gender'].str.lower().str.replace(" ", "").replace({"fe male": "female"})
        self.x['loan_intent'] = self.x['loan_intent'].replace({'DEBTCONSOLIDATION': 'DEBT CONSOLIDATION', 'HOMEIMPROVEMENT': 'HOME IMPROVEMENT'})
        self.x['person_gender'] = self.x['person_gender'].map({'Male': 1, 'Female': 0})
        self.x['previous_loan_defaults_on_file'] = self.x['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

    def encode_education(self):
        person_education_encoder = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
        self.x['person_education'] = self.x['person_education'].replace(person_education_encoder)

    def one_hot_encode(self):
        home_ownership_encoder = OneHotEncoder()
        loan_intent_encoder = OneHotEncoder()

        home_ownership_train = pd.DataFrame(home_ownership_encoder.fit_transform(self.x[['person_home_ownership']]).toarray(),
                                             columns=home_ownership_encoder.get_feature_names_out())
        loan_intent_train = pd.DataFrame(loan_intent_encoder.fit_transform(self.x[['loan_intent']]).toarray(),
                                         columns=loan_intent_encoder.get_feature_names_out())

        self.x = pd.concat([self.x, home_ownership_train, loan_intent_train], axis=1)
        self.x = self.x.drop(['person_home_ownership', 'loan_intent'], axis=1)

    def scale_data(self):
        numerical = self.x.select_dtypes(include=['number']).columns.tolist()
        self.scalers = {}
        for col in numerical:
            scaler = RobustScaler()
            self.x[[col]] = scaler.fit_transform(self.x[[col]])
            self.scalers[col] = scaler
            pickle.dump(scaler, open(f"{col}_scaler.pkl", "wb"))

    def preprocess(self):
        self.handle_missing_values()
        self.handle_categorical_data()
        self.encode_education()
        self.one_hot_encode()
        self.scale_data()

class ModelTrainer:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.rf_model = RandomForestClassifier(random_state=42, criterion='gini', max_depth=4, n_estimators=100)
        self.xgb_model = XGBClassifier(random_state=42, n_estimators=100, min_child_weight=50, max_depth=8)

    def train_rf(self):
        self.rf_model.fit(self.x_train, self.y_train)
        return self.rf_model

    def train_xgb(self):
        self.xgb_model.fit(self.x_train, self.y_train)
        return self.xgb_model

class ModelEvaluator:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        return classification_report(self.y_test, y_pred)

class ModelSaver:
    def __init__(self):
        pass

    def save_model(self, model, filename):
        pickle.dump(model, open(filename, "wb"))

    def save_encoder(self, encoder, filename):
        pickle.dump(encoder, open(filename, "wb"))

class LoanPrediction:
    def __init__(self, df):
        self.df = df
        self.preprocessor = DataPreprocessor(df)
        self.model_trainer = None
        self.model_evaluator = None
        self.model_saver = ModelSaver()

    def prepare_data(self):
        self.preprocessor.preprocess()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.preprocessor.x, self.preprocessor.y, test_size=0.2, random_state=42)

    def train_models(self):
        self.model_trainer = ModelTrainer(self.x_train, self.y_train)
        self.rf_model = self.model_trainer.train_rf()
        self.xgb_model = self.model_trainer.train_xgb()

    def evaluate_models(self):
        self.model_evaluator = ModelEvaluator(self.rf_model, self.x_test, self.y_test)
        rf_report = self.model_evaluator.evaluate()

        self.model_evaluator = ModelEvaluator(self.xgb_model, self.x_test, self.y_test)
        xgb_report = self.model_evaluator.evaluate()

        print("Random Forest Evaluation:\n", rf_report)
        print("XGBoost Evaluation:\n", xgb_report)

    def save_best_model(self):
        self.model_saver.save_model(self.xgb_model, "xgb_model.pkl")
        print("Model XGBoost saved as xgb_model.pkl")


# Main script to run the loan prediction pipeline
df = pd.read_csv('Dataset_A_loan.csv')
loan_prediction = LoanPrediction(df)
loan_prediction.prepare_data()
loan_prediction.train_models()
loan_prediction.evaluate_models()
loan_prediction.save_best_model()

