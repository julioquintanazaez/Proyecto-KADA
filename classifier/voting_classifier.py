import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import pytz
# Classifiers stuff
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
# Custom classifier
from classifier.combo_classifier import ComboClassifier
# Classification stuff
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Utils
from vectors.custom_vectorizer import CustomTfidfVectorizer
from utils.text_processor import TextProcessor

class Voting_ProductClassifier:
    def __init__(self):
        self.vectorizer = CustomTfidfVectorizer(strip_accents='unicode')
        self.text_processor = TextProcessor()
        self.combo_classifier = ComboClassifier()
        self.confusion_matrix = []
        self.tags = []
        self.model = self.load_latest_model()
        self.vectorizer = self.load_latest_vectorizer()  # Cargar el vectorizador al iniciar

    def set_tags(self, y_tags):
        self.tags = np.unique(y_tags)

    def load_latest_model(self):
        model_folder = os.path.join('train_folder', 'train_saves_voting')
        model_files = [f for f in os.listdir(model_folder) if f.endswith('.pkl')]
        if not model_files:
            return None
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_folder, x)))
        print(model_files[-1])
        model = joblib.load(os.path.join(model_folder, model_files[-1]))
        return model

    def load_latest_vectorizer(self):
        vectorizer_folder = os.path.join('train_folder', 'train_saves_vectorizer')
        vectorizer_files = [f for f in os.listdir(vectorizer_folder) if f.endswith('.pkl')]
        if not vectorizer_files:
            return CustomTfidfVectorizer(strip_accents='unicode')  # Retornar un nuevo vectorizador si no existe
        vectorizer_files.sort(key=lambda x: os.path.getmtime(os.path.join(vectorizer_folder, x)))
        print(vectorizer_files[-1])
        vectorizer = joblib.load(os.path.join(vectorizer_folder, vectorizer_files[-1]))
        return vectorizer

    def train_model(self, excel_file):
        self.df_train = pd.read_excel(excel_file)
        self.df_train['process_name'] = self.df_train['name'].apply(self.text_processor.text_process)
        X = self.df_train['process_name']
        y = self.df_train['tag']
        self.set_tags(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        
        if self.model is None:
            clf1 = MultinomialNB()
            clf2 = RandomForestClassifier(random_state=1)
            clf3 = SVC(probability=True, random_state=1)
            voting_clf = VotingClassifier(estimators=[
                ('lr', clf1),
                ('dt', clf2),
                ('svc', clf3)
            ], voting='soft')
            self.model = voting_clf
            self.model.fit(X_train_tfidf, y_train)
            self.save_model()
            self.save_vectorizer()  # Guardar el vectorizador después de entrenar
        else:
            self.model.fit(X_train_tfidf, y_train)

        y_pred = self.model.predict(X_val_tfidf)
        self.confusion_matrix = confusion_matrix(y_val, y_pred)
        print(classification_report(y_val, y_pred))


    def save_model(self):
        cuba_tz = pytz.timezone('America/Havana')
        current_time_cuba = datetime.now(cuba_tz)
        timestamp = current_time_cuba.strftime('%Y%m%d_%H%M%S')
        model_folder = os.path.join('train_folder', 'train_saves_voting')
        new_model_filename = os.path.join(model_folder, f'model_voting_{timestamp}.pkl')
        joblib.dump(self.model, new_model_filename)

    def save_vectorizer(self):
        cuba_tz = pytz.timezone('America/Havana')
        current_time_cuba = datetime.now(cuba_tz)
        timestamp = current_time_cuba.strftime('%Y%m%d_%H%M%S')
        vectorizer_folder = os.path.join('train_folder', 'train_saves_vectorizer')
        new_vectorizer_filename = os.path.join(vectorizer_folder, f'vectorizer_{timestamp}.pkl')
        joblib.dump(self.vectorizer, new_vectorizer_filename)

    def retrain_model(self, new_data):
        model_folder = os.path.join('train_folder', 'train_saves_voting')
        new_data['process_name'] = new_data['name'].apply(self.text_processor.text_process)

        df_combined = pd.concat([self.df_train, new_data], ignore_index=True)
        X_combined = df_combined['process_name']
        y_combined = df_combined['tag']
        self.set_tags(y_combined)
        X_combined_tfidf = self.vectorizer.transform(X_combined)  # Usar el vectorizador cargado

        clf1 = MultinomialNB()
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = SVC(probability=True, random_state=1)
        voting_clf = VotingClassifier(estimators=[
            ('lr', clf1),
            ('dt', clf2),
            ('svc', clf3)
        ], voting='soft')
        self.model = voting_clf
        self.model.fit(X_combined_tfidf, y_combined)
        y_pred = self.model.predict(X_combined_tfidf)
        self.confusion_matrix = confusion_matrix(y_combined, y_pred)
        print("Métricas de evaluación del modelo después del reentrenamiento:")
        print(classification_report(y_combined, y_pred))
        self.save_model()
        df_new = df_combined.copy()
        df_new = df_new[['id', 'name', 'current_price', 'tag']]
        cuba_tz = pytz.timezone('America/Havana')
        current_time_cuba = datetime.now(cuba_tz)
        timestamp = current_time_cuba.strftime('%Y%m%d_%H%M%S')
        combined_excel_file = os.path.join(model_folder, f'train_products_{timestamp}.csv')
        df_new.to_csv(combined_excel_file, index=False)
        print(f"Datos combinados guardados en {combined_excel_file}")
        return True


    def predict_tags(self, df_test):
        df_test['process_name'] = df_test['name'].apply(self.text_processor.text_process)

        X_test_tfidf = self.vectorizer.transform(df_test['process_name'])
        probabilities = self.model.predict_proba(X_test_tfidf)
        threshold = 0.6
        tags = []
        for prob in probabilities:
            if max(prob) < threshold:
                tags.append('otros')
            else:
                tags.append(self.model.classes_[prob.argmax()])
        df_test['tag'] = tags
        df_predict = df_test.copy()
        df_predict.loc[:, 'tag'] = df_predict.apply(self.combo_classifier.combo_clf, axis=1)
        cuba_tz = pytz.timezone('America/Havana')
        current_time_cuba = datetime.now(cuba_tz)
        formatted_time = current_time_cuba.strftime('%d/%m/%Y %H:%M:%S')
        df_predict['tag_updated_at'] = formatted_time
        return df_predict[['id', 'name', 'current_price', 'tag', 'tag_updated_at']]


    def get_cm(self):
        return self.confusion_matrix, self.tags


    def get_tags(self):
        return self.tags

