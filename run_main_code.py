import os
import pandas as pd
#from dotenv import  load_dotenv

from db.db_conector import DatabaseConnector
from classifier.voting_classifier import Voting_ProductClassifier as VPClassifier

# Create your classes here

print("Entro al script")

# Archivo Excel que contiene los datos de entrenamiento
excel_file = './otros.xlsx'

print("Leyendo datos de entrenamiento")

# Crear una instancia del clasificador de productos
clf_voting = VPClassifier()

print("Entrenando el clasificador")

# Entrenar el modelo, como parámetro podemos pasar el modelo que deseemos 
clf_voting.train_model(excel_file)

print("Leyendo datos de prueba")

# Archivo Excel que contiene los datos de prueba
df_test = pd.read_excel('juego de datos.xlsx')

print("Realizando predicciones")

# Realizar las predicciones de las etiquetas
df_predictions = clf_voting.predict_tags(df_test)
 

#df_predictions['tag_updated_at'] = pd.to_datetime(df_predictions['tag_updated_at'])
df_predictions.to_csv('products_predictions_test.csv', index=False)

#df_new = pd.read_excel('otros.xlsx')
#clf_voting.retrain_model(df_new)


"""       
    
# Supongamos que tienes un DataFrame de prueba para hacer predicciones
conector = DatabaseConnector(read_root())

rows = ['id', 'name', 'current_price','tag','shop_id','description']
#df_test = conector.data_postgresql('product', rows)
#print(df_test)

df_test = pd.read_excel('juego de datos.xlsx')
df_predictions = classifier.predict_tags(df_test)
#df_predictions['tag_updated_at'] = pd.to_datetime(df_predictions['tag_updated_at'])
#df_predictions.to_csv('products_predictions_test.csv', index=False)


table_name = 'product'
column_name = 'tag'
column_name2 = 'tag_updated_at'


#conector.update_rowP(df_predictions, table_name, column_name2)

#'2024-11-12 19:15:38+00:00'
#'11/12/2024 1:56:39 PM'

df_retrain = conector.data_postgresql_filtered_by_date(table_name, rows,column_name,'2024-11-12 19:15:38+00:00')

#df_retrain.to_csv('recomendador.csv', index= False)
#print(df_retrain)
#classifier.retrain_model(df_retrain)

recomender = Recomender(df_retrain)
recomender.train_recomender()

print(recomender.get_recommendations(['Sal común Cubanacan (1 kg / 2.2 lb)','Aceite de oliva extra virgen Medeiros (502 ml)'
                                         ,'Café molido Guantanamera (1 kg / 2.2 lb)']))

                                         
                                         """