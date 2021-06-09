import datetime
from dateutil.relativedelta import relativedelta
import os

import pandas as pd
import sqlalchemy
from tqdm import tqdm

DATA_PATH = 'data/olist_dsa.db'
QUERY_PATH = 'sql/Script_ABT_olist_dtref_safra_20200818.sql'

primeira_safra = "2018-04-01"
ultima_safra = "2018-06-01"

def get_abt(DATA_PATH,QUERY_PATH,primeira_sfra,ultima_safra):

    with open( QUERY_PATH, 'r' ) as open_file:
        query = open_file.read()

    primeira_safra_datetime = datetime.datetime.strptime("2017-03-01", "%Y-%m-%d")
    ultima_safra_datetime = datetime.datetime.strptime("2018-06-01", "%Y-%m-%d")

    con = sqlalchemy.create_engine( 'sqlite:///' + DATA_PATH )

    def exec_etl( queries, con ):
        for q in tqdm(queries.split(";")[:-1]):
            con.execute(q)

    while primeira_safra_datetime <= ultima_safra_datetime:
        safra = primeira_safra_datetime.strftime("%Y-%m-%d")
        query_exec = query.format( data_ref=safra )
        primeira_safra_datetime = primeira_safra_datetime + relativedelta(months=+1)
        
        if safra == primeira_safra:
            query_exec += f'''DROP TABLE IF EXISTS TB_ABT;
                                CREATE TABLE TB_ABT AS
                                SELECT '{safra}' as dt_ref,* FROM vw_olist_abt_p2;'''
        else:
            query_exec += f'''INSERT INTO TB_ABT
                                SELECT '{safra}' as dt_ref,* FROM vw_olist_abt_p2;'''
        print(safra)
        exec_etl(query_exec, con)
        





if __name__ == '__main__':
    get_abt(DATA_PATH,QUERY_PATH,primeira_safra,ultima_safra)