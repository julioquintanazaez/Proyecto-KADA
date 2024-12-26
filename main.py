from  fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse

import pandas as pd
import os
import pytz
from datetime import datetime

from classifier.classifier import ProductClassifier
from classifier.voting_classifier import Voting_ProductClassifier as VPClassifier
from db.db_conector import DatabaseConnector
from recomender.recomender import Recomender

from schemas.matrix_response import MatrixResponse, ListResponse
from typing import List

from dotenv import load_dotenv

load_dotenv()

# Empty classifier
clf_voting = None
db_config = {}

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
        excel_file = './otros.xlsx'
        clf_voting.train_model(file)
        print("Salio al entrenamiento")
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
async def retrain_model():
   
    # Se debe leer de la base de datos y crear el dataframe con datos nuevos
    conector = DatabaseConnector(db_config)
    table_name = 'product'
    column_name = 'tag'
    rows_query = ['id', 'name', 'description', 'tag']

    """Que fecha se toma para el reentreno del algoritmo, logicamente no debe ser un valor fijo
    ahora, es una valor pasado por parámetros, o se detecta automáticamente.

    Estuve mirando la lógica de la consulta filtro y escoges los que son distintos de "otros" por lo que
    deduje que el valor de la variable "column_name" en este caso es "tag".  
    """
    df_new = conector.data_postgresql_filtered_by_date(table_name, rows_query, column_name, '2024-11-12')  #2024-11-12 19:15:38+00:00
    print(df_new.shape)
    print(df_new.columns)
    print(df_new.head(5))
    #df_new = pd.read_excel('otros.xlsx')   # Se comenta una vez conectada la db

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
    cm, tags = clf_voting.get_cm()
    print(tags)
    return cm


@app.get("/get_tags", response_model=List[str])  
async def get_tags():
   
    # Return the confusion matrix
    tags = clf_voting.get_tags()

    return tags
    

  
