from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import json
from pprint import pprint
import pandas as pd

# Definir las palabras prioritarias y sus pesos
''' lo  q se quier es este diccionario q esta por aqui convertirlo en una bd de datos en este 
caso un archivo json y de alli sacar el diccionario y usarlo en el codigo, para no tener q modificar el codigo
cada vez  q se añada una etiqueta sea solamente añadirlo al archivo json y listo    '''
priority_words_temp = {
    'cafe': 2.0,
    'huevo': 2.0,
    'arroz': 2.0,
    'azucar': 2.0,
    'frijol' : 2.0,
    'frijoles': 2.0,
    'aceite': 2.0,
    'huevos': 2.0,
    'pan': 2.0,
    'sal': 2.0,
}

class CustomTfidfVectorizer(TfidfVectorizer):
    def __init__(self):
        self.priority_words = self.load_priority_words_from_json()  # Cargar priority words from json al iniciar


    def load_priority_words_from_json(self):
        """This method read the priority words from a file. 
        If the file doesn't exists it reae from a temporal priority words.
        The method return the priority_words as a dictionary 
        """
        filename = "priority_words.json"
        priority_folder = os.path.join('vectors', 'priority_words_tags')
        priority_words_tags_file = os.path.join(priority_folder, filename)
        if not priority_words_tags_file:
            return priority_words_temp  # Retorna las words temporales
        try:
            with open(priority_words_tags_file) as f:
                priority_words = json.load(f)
                #pprint(priority_words)
                return priority_words["priority_words"]
        except:
            print(f"Cant read the file in the folder {priority_words_tags_file}")
            return priority_words_temp  
    

    def fit_transform(self, raw_documents, y=None):
        # Call the parent method to get the original tf-idf matrix
        tfidf_matrix = super().fit_transform(raw_documents, y)

        # Get the feature names
        feature_names = self.get_feature_names_out()

        # Create a weight array
        weights = tfidf_matrix.toarray()

        # Apply custom weights to the relevant features
        for word, weight in self.priority_words.items():
            if word in feature_names:
                index = feature_names.tolist().index(word)
                weights[:, index] *= weight

        # Return the modified tf-idf matrix
        return weights

    def transform(self, raw_documents):
        tfidf_matrix = super().transform(raw_documents)
        feature_names = self.get_feature_names_out()
        weights = tfidf_matrix.toarray()

        for word, weight in self.priority_words.items():
            if word in feature_names:
                index = feature_names.tolist().index(word)
                weights[:, index] *= weight

        return weights




