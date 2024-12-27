from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Definir las palabras prioritarias y sus pesos
''' lo  q se quier es este diccionario q esta por aqui convertirlo en una bd de datos en este 
caso un archivo json y de alli sacar el diccionario y usarlo en el codigo, para no tener q modificar el codigo
cada vez  q se añada una etiqueta sea solamente añadirlo al archivo json y listo    '''
priority_words = {
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
    def fit_transform(self, raw_documents, y=None):
        # Call the parent method to get the original tf-idf matrix
        tfidf_matrix = super().fit_transform(raw_documents, y)

        # Get the feature names
        feature_names = self.get_feature_names_out()

        # Create a weight array
        weights = tfidf_matrix.toarray()

        # Apply custom weights to the relevant features
        for word, weight in priority_words.items():
            if word in feature_names:
                index = feature_names.tolist().index(word)
                weights[:, index] *= weight

        # Return the modified tf-idf matrix
        return weights

    def transform(self, raw_documents):
        tfidf_matrix = super().transform(raw_documents)
        feature_names = self.get_feature_names_out()
        weights = tfidf_matrix.toarray()

        for word, weight in priority_words.items():
            if word in feature_names:
                index = feature_names.tolist().index(word)
                weights[:, index] *= weight

        return weights




