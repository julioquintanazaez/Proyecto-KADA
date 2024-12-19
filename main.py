import pandas as pd

from classifier.classifier import ProductClassifier
from db.db_conector import DatabaseConnector
from recomender.recomender import Recomender
from  fastapi import FastAPI
import os
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    host = os.environ.get("HOST")
    database = os.environ.get("DATABASE")
    user = os.environ.get("USER")
    password = os.environ.get("PASSWORD")
    db_config = {
        'host': host,
        'database': database,
        'user': user,
        'password': password
    }
    return db_config






# Archivo Excel que contiene los datos de entrenamiento
excel_file = 'otros.xlsx'

# Crear una instancia del clasificador de productos
classifier = ProductClassifier(excel_file)

# Entrenar el modelo
classifier.train_model()

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
