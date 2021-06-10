import datetime
import sys

from abt import get_abt
from train import get_models

DATA_PATH = 'data/olist_dsa.db'
QUERY_PATH = 'sql/Script_ABT_olist_dtref_safra_20200818.sql'
model_path='models/'

primeira_safra = "2018-04-01"
ultima_safra = "2018-06-01"

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    
    if '-abtonly' in sys.argv:
        get_abt(DATA_PATH,QUERY_PATH,primeira_safra,ultima_safra)
    elif '-trainonly' in sys.argv:
        campeao_nome=get_models(DATA_PATH, model_path)
    else:
        get_abt(DATA_PATH,QUERY_PATH,primeira_safra,ultima_safra)
        campeao_nome=get_models(DATA_PATH, model_path)

    end_time = datetime.datetime.now()

    print('================================================')
    print('\nScript completo.')
    print(f'Modelo campeão {campeao_nome[0]} AUC {campeao_nome[1]} \n')
    print(f'O tempo total foi {end_time - start_time}\n')
    print('================================================')

