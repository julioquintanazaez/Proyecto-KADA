import os
import pandas as pd
from dotenv import  load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import time
import pandas as pd
from datetime import datetime
from db.db_conector import DatabaseConnector
from classifier.voting_classifier import Voting_ProductClassifier as VPClassifier


#pip install APScheduler Julio este es el package q tienes q instalar nuevo para las tareas programadas

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
 

df_predictions['tag_updated_at'] = pd.to_datetime(df_predictions['tag_updated_at'])
df_predictions.to_csv('products_predictions_test.csv', index=False)

df_new = pd.read_excel('otros.xlsx')
df_new_cleaned = df_new.dropna(subset=['id', 'name', 'current_price' , 'tag'])
clf_voting.retrain_model(excel_file,df_new)

load_dotenv()
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

db_config = db_start()

conector = DatabaseConnector(db_start())
df_test_unique_tags= conector.get_unique_tags()


rows_query = ['id', 'name', 'current_price', 'shop_id','description','tag']
df_test = conector.data_postgresql_empty_tag('product', rows_query)




def classify_empty_tags():
    conector = DatabaseConnector(db_config)
    empty_tag_products = conector.data_postgresql_empty_tag('product', ['id', 'name', 'current_price,tag'])

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





# Función para reentrenar el modelo con la fecha más reciente

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
        # Convertir most_recent_date a cadena en un formato legible
        most_recent_date_str = most_recent_date.strftime('%Y-%m-%d %H:%M:%S')

        # Verificar si la fecha ya ha sido registrada
        if most_recent_date_str in logged_dates:
            print("Ya se realizó el entrenamiento con estos datos.")
            return

        df_new = conector.data_postgresql_filtered_by_date('product', ['id', 'name', 'current_price', 'tag'], 'tag',
                                                           most_recent_date)



        if df_new.empty:
            print("No hay datos nuevos para reentrenar.")
            return

        # Llamar al método de reentrenamiento
        clf_voting.retrain_model(excel_file, df_new)
        print("Reentrenamiento realizado.")

        # Guardar la fecha más reciente en el archivo de log
        with open(log_file, 'a') as file:
            file.write(most_recent_date_str + '\n')
    else:
        print("No se pudo obtener la fecha más reciente.")

# Crear el programador
scheduler = BackgroundScheduler()

# Agregar tareas programadas (cada 1 minuto)
#scheduler.add_job(classify_empty_tags, 'interval', minutes=1, id='classify_job', replace_existing=True)
#scheduler.add_job(retrain_model_with_recent_date, 'interval', minutes=1, id='retrain_job', replace_existing=True)

# Iniciar el programador
scheduler.start()
print("Scheduler iniciado. Presiona Ctrl+C para detenerlo.")

try:
    # Mantener el script en ejecución
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    # Detener el programador al salir
    scheduler.shutdown()
    print("Scheduler detenido.")










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