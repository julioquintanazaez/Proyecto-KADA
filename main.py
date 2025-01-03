from  fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler

import pandas as pd
import os
import pytz
from datetime import datetime

from classifier.classifier import ProductClassifier
from classifier.voting_classifier import Voting_ProductClassifier as VPClassifier
from db.db_conector import DatabaseConnector
from recomender.recomender import Recomender

from vectors.custom_vectorizer import CustomTfidfVectorizer

from schemas.validators import DateTimeModel

from typing import List

from dotenv import load_dotenv

load_dotenv()

# Empty classifier
clf_voting = None
db_config = {}

excel_file = './otros.xlsx'

# Carpeta donde se guardarán los archivos
UPLOAD_DIRECTORY = "train_folder/train_files"

# Crear la carpeta si no existe
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app = FastAPI()

def db_start():
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
 

def train_model_with_file(file):    

    # Train the classifier
    try:
        print("Entro al entrenamiento")
        # Crear una instancia del clasificador de productos
        #excel_file = './otros.xlsx'
        clf_voting.train_model(file)
        print("Training done successfuly....")
        return JSONResponse(status_code=200, content="Training successful...")

    except Exception as e:        
        return HTTPException (status_code=404, detail="Fail training the classifier...")  


@app.on_event("startup")
async def startup_event():
    try:
        global clf_voting
        clf_voting = VPClassifier()
       
        global db_config
        db_config = db_start()

        # Crear el programador
        scheduler = BackgroundScheduler()
        # Agregar tareas programadas (cada día)
        scheduler.add_job(classify_empty_tags, 'interval', days=1, id='classify_job', replace_existing=True)
        scheduler.add_job(retrain_model_with_recent_date, 'interval', days=1, id='retrain_job', replace_existing=True)
        # Iniciar el programador
        scheduler.start()

    except Exception as e:
        return HTTPException (status_code=404, detail="Fail database conection...")  


@app.get("/")
def index():
    return "App for Classifier and Recomender stuff"


@app.post("/uploadfile_and_trainmodel/", response_class=JSONResponse)
async def uploadfile_and_trainmodel(file: UploadFile = File(...)):

    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)

    # Guardar el archivo
    with open(file_location, "wb") as file_object:
        file_object.write(await file.read())            

    # Train the classifier
    train_model_with_file(file_location)
    
    return JSONResponse(status_code=200, content="Training successful...")


@app.get("/predict_tags", response_class=JSONResponse)  
async def predict_tags():

    cuba_tz = pytz.timezone('America/Havana')
    current_time_cuba = datetime.now(cuba_tz)
    timestamp = current_time_cuba.strftime('%Y%m%d_%H%M%S')
    
    if db_config != {}:
        # Supongamos que tienes un DataFrame de prueba para hacer predicciones
        try:
            conector = DatabaseConnector(db_config)
            rows_query = ['id', 'name', 'current_price', 'shop_id','description']
            df_test = conector.data_postgresql('product', rows_query)
            # Make predictions
            #df_test = pd.read_excel('juego de datos.xlsx')
            df_predictions = clf_voting.predict_tags(df_test)
            df_predictions['tag_updated_at'] = pd.to_datetime(df_predictions['tag_updated_at'], dayfirst=True) #, dayfirst=True
            df_predictions.to_csv("datos etiquetados.csv") #Borrar cuando este la validación de etiquetas
            # Update database with predictions
            """Revisar esta lógica, me genera dudas, no pueden ser las dos columnas
            actualizadas a la vez.
            """
            table_name = 'product'
            column_name = 'tag' 
            column_name2 = 'tag_updated_at'
            conector.update_rowP(df_predictions, table_name, column_name) 
            conector.update_rowP(df_predictions, table_name, column_name2)        
        
            return JSONResponse(status_code=200, content="Tags prediction process successful...")
        
        except:
            return HTTPException (status_code=404, detail="Fail database conection...")  
    else:        
        return HTTPException (status_code=404, detail="Fail database config...")  


@app.get("/retrain_model", response_class=JSONResponse)  
async def retrain_model(date):  #Le agregué un validador de datos estan en la carpeta shcemas
   
    # Se debe leer de la base de datos y crear el dataframe con datos nuevos
    conector = DatabaseConnector(db_config)
    table_name = 'product'
    column_name = 'tag'
    rows_query = ['id', 'name', 'description', 'tag']

    df_new = conector.data_postgresql_filtered_by_date(table_name, rows_query, column_name, date)  #2024-11-12 19:15:38+00:00
    print(df_new.shape)
    print(df_new.columns)
    print(df_new.head(5))

    df_new.to_csv('filtered_datos.xlsx')   # Se comenta una vez conectada la db
    
    try:
        print("Entro al try")
        clf_voting.retrain_model(df_new)
        print("paso la clasificación")

        return JSONResponse(status_code=200, content="Retraining successful...")
    
    except:
        return HTTPException (status_code=404, detail="Fail retaining classifier algorithm...")  


@app.get("/get_predictions_matrix", response_model=List[List[int]])  #response_model=List[List[int]]
async def get_predictions_matrix():
   
    # Return the confusion matrix
    cm, cm_tags = clf_voting.get_cm()

    return cm


"""Esto lo puse para testear que estabamos leyendo las prioritywords
"""
@app.get("/test_stuff")  
async def test_stuff():

    cvzer = CustomTfidfVectorizer()
    print(cvzer.priority_words)

    try:
        retrain_model_with_recent_date()
    except TypeError as e:
        return HTTPException (status_code=404, detail=e)  

    return "Correct"

'''Otra cosa no se si llegaste a ejecutar el reentrenamiento con la logica de abrir desde un json,
 me devuelve un error tal q no esta inicializada el atributo q inicializas
y entonces da error en qnto puedas revisa eso por ahora lo puse a funcionar con los temporales
'''
def classify_empty_tags():
    conector = DatabaseConnector(db_config)
    empty_tag_products = conector.data_postgresql_empty_tag('product', ['id', 'name', 'current_price', 'tag','tag_updated_at'])
    if empty_tag_products.empty:
        print("Todos los productos están clasificados.")
    else:
        df_predictions = clf_voting.predict_tags(empty_tag_products)
        table_name = 'product'
        column_name = 'tag'
        column_name2 = 'tag_updated_at'
        conector.update_rowP(df_predictions, table_name, column_name)
        conector.update_rowP(df_predictions, table_name, column_name2)
        print("Clasificación realizada.")

def retrain_model_with_recent_date():
    conector = DatabaseConnector(db_config)
    log_file = 'training_log.txt'
    # Leer las fechas del archivo de log
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            logged_dates = file.read().splitlines()
    else:
        logged_dates = []
    most_recent_date = conector.get_most_recent_tag_updated_at('product')
    if most_recent_date:
        most_recent_date_str = most_recent_date.strftime('%Y-%m-%d %H:%M:%S')
        if most_recent_date_str in logged_dates:
            print("Ya se realizó el entrenamiento con estos datos.")
            return
        df_new = conector.data_postgresql_filtered_by_date('product', ['id', 'name', 'current_price', 'tag'], 'tag', most_recent_date)
        if df_new.empty:
            print("No hay datos nuevos para reentrenar.")
            return
        clf_voting.retrain_model(excel_file, df_new)
        print("Reentrenamiento realizado.")
        with open(log_file, 'a') as file:
            file.write(most_recent_date_str + '\n')
    else:
        print("No se pudo obtener la fecha más reciente.")




    

  
