import os
import pandas as pd
#from dotenv import  load_dotenv

from db.db_conector import DatabaseConnector
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
#from classifier.classifier import ProductClassifier
from classifier.base_classifier import ProductClassifier as PCF

# Create your classes here

print("Entro al script")

# Archivo Excel que contiene los datos de entrenamiento
excel_file = './otros.xlsx'

print("Leyendo datos de entrenamiento")

# Crear una instancia del clasificador de productos
clf_Multinominal = PCF(excel_file)
clf_ExtraTree = PCF(excel_file)
clf_LinearSVC = PCF(excel_file)

print("Entrenando el clasificador")

# Entrenar el modelo, como parámetro podemos pasar el modelo que deseemos 
clf_Multinominal.train_model(base_model=MultinomialNB())
clf_ExtraTree.train_model(base_model=ExtraTreeClassifier())
clf_LinearSVC.train_model(base_model=RandomForestClassifier(n_estimators=100, random_state=42))

print("Leyendo datos de prueba")

# Archivo Excel que contiene los datos de prueba
#df_test = pd.read_excel('juego de datos.xlsx')

print("Realizando predicciones")

# Realizar las predicciones de las etiquetas
#df_predictions = clf_Multinominal.predict_tags(df_test)


#df_predictions['tag_updated_at'] = pd.to_datetime(df_predictions['tag_updated_at'])
#df_predictions.to_csv('products_predictions_test.csv', index=False)

