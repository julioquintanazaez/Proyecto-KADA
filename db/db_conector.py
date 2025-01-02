import pandas as pd
from sqlalchemy import create_engine, inspect, text

'''' ahora agregue 2 consultas nuevas todavia  una q devuelva un dataframe de productos si el tag esta vacio y otro  
q me devuelva la fecha de update mas reciente, para automatizar reentreno y la clasificacion, yo veo de la siguiente 
de esta forma si queremos 1ro ejecutamos la clasificacion normal y ya luego se deja activa la automatica'''


class DatabaseConnector:
    def __init__(self, db_config):
        self.db_config = db_config

    def connect(self):
        connection_string = f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
        engine = create_engine(connection_string)
        return engine

    def data_postgresql(self, table, rows):
        engine = self.connect()
        rows_str = ', '.join(rows)
        query = f"SELECT {rows_str} FROM {table};"
        try:
            df = pd.read_sql_query(query, engine)
            print("Datos extraídos exitosamente")
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")
            df = None
        return df

    def update_rowP(self, dataframe, table, column_name):
        engine = self.connect()
        inspector = inspect(engine)
        if column_name not in [col['name'] for col in inspector.get_columns(table)]:
            print(f"La columna '{column_name}' no existe en la tabla '{table}'.")
            return
        try:
            with engine.connect() as connection:
                with connection.begin():
                    for index, row in dataframe.iterrows():
                        value = row[column_name]
                        id_value = row['id']
                        current_value = connection.execute(
                            text(f"SELECT {column_name} FROM {table} WHERE id = :id;"),
                            {'id': id_value}
                        ).scalar()
                        if current_value != value:
                            connection.execute(
                                text(f"UPDATE {table} SET {column_name} = :value WHERE id = :id;"),
                                {'value': value, 'id': id_value}
                            )
                        else:
                            print(f"No se requiere actualización para id {id_value}: el valor ya es {current_value}")
            print(f"Column '{column_name}' successfully updated in the table '{table}'")
        except Exception as e:
            print(f"Error updating column: {e}")

    def data_postgresql_filtered_by_date(self, table_name, columns, filter_column, specific_date):
        engine = self.connect()
        columns_str = ', '.join(columns)
        query = f"""
        SELECT {columns_str} 
        FROM {table_name} 
        WHERE {filter_column} != 'otros' 
        AND DATE(tag_updated_at) = '{specific_date}';
        """
        try:
            df = pd.read_sql_query(query, engine)
            print("Data extracted successfully with filters applied")
        except Exception as e:
            print(f"Error executing the query: {e}")
            df = None
        return df

    def data_postgresql_empty_tag(self, table, columns):
        engine = self.connect()
        columns_str = ', '.join(columns)
        query = f"""
        SELECT {columns_str} 
        FROM {table} 
        WHERE tag IS NULL OR tag = '';
        """
        try:
            df = pd.read_sql_query(query, engine)
            print("Datos extraídos exitosamente con el filtro de tag vacío")
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")
            df = None
        return df

    def get_most_recent_tag_updated_at(self, table):
        engine = self.connect()
        query = f"""
        SELECT MAX(tag_updated_at) AS most_recent_date 
        FROM {table};
        """
        try:
            with engine.connect() as connection:
                result = connection.execute(text(query)).scalar()
            print("Fecha más reciente extraída exitosamente")
        except Exception as e:
            print(f"Error al ejecutar la consulta: {e}")
            result = None
        return result