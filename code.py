import os
print(os.sys.version)

my_name = 'code'

for i in [1,2,3,4]:
   print([i])


import pandas as pd
import numpy as np
import feature_engine.missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport
import sqlalchemy

DATA_PATH = 'dados/olist_dsa.db'
con = sqlalchemy.create_engine( "sqlite:///"+DATA_PATH )

    
df = pd.read_sql_table( "TB_ABT", con )

target = 'fl_venda'
to_remove = ['seller_id', 'seller_city', 'seller_zip_code_prefix', 'dt_ref'] + [target]
cat_vars = ['seller_state']
num_vars = list(set(df.columns.tolist()) - set( to_remove ) - set(cat_vars))

tend_var = [i for i in num_vars if i.startswith("tend")]
qtd_vars = [i for i in num_vars if i.startswith("qtd") or i.startswith("quant")]
media_vars = [i for i in num_vars if i.startswith("media")]
max_vars = [i for i in num_vars if i.startswith("max")]
dias = [i for i in num_vars if i.startswith("dias")]
prop = [i for i in num_vars if i.startswith("prop")]

df['dt_ref'] = pd.to_datetime(df['dt_ref']) # Convertendo para tipo datetime
max_dt = df['dt_ref'].max() # Pegando a data máxima
df_oot = df[df['dt_ref']==max_dt].copy() # Separando para base com maior datas
df_train = df[df['dt_ref'] < max_dt].copy() # Separando para base menor que a maior data

X_train, X_test, y_train, y_test = model_selection.train_test_split( df_train[num_vars+cat_vars],
                                                                     df_train[target],
                                                                     random_state=1992,
                                                                     test_size=0.25)

imputer_zero = mdi.ArbitraryNumberImputer( arbitrary_number=0,
                                           variables=qtd_vars+media_vars+max_vars+dias+prop)

imputer_um = mdi.ArbitraryNumberImputer( arbitrary_number=1,
                                         variables=tend_var)

onehot = ce.OneHotCategoricalEncoder(variables=cat_vars, drop_last=True) # Cria Dummys

model = linear_model.LogisticRegression(penalty='l1', solver='liblinear' ) # Definição do modelo

full_pipeline = Pipeline( steps=[
    ('zero', imputer_zero),
    ('um', imputer_um),
    ("onehot", onehot),
    ('model', model) ] )

param_grid = { 'model__C':[0.0167, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1], # linspace
               'model__random_state':[1992]}

search = model_selection.GridSearchCV(full_pipeline,
                                       param_grid,
                                       cv=5,
                                       n_jobs=-1,
                                       scoring='roc_auc')

search.fit(X_train, y_train) # Executa o treinamento!!

best_model = search.best_estimator_

cv_result = pd.DataFrame(search.cv_results_) # Pega resultdos do grid
cv_result = cv_result.sort_values(by='mean_test_score', ascending = False,)

y_test_prob = best_model.predict_proba(X_test)[:, 1]
auc_test = metrics.roc_auc_score( y_test, y_test_prob)
print( "Área sob a curva ROC:", auc_test)

best_model.fit( df_train[num_vars+cat_vars], df_train[target] )

# Verificando erro na base de teste
y_test_prob = best_model.predict_proba(df_oot[ num_vars+cat_vars ])[:, 1]
auc_oot = metrics.roc_auc_score( df_oot[target], y_test_prob)
print( "Área sob a curva ROC:", auc_oot)

best_model.fit(df[num_vars+cat_vars], df[target])

features = best_model[:-1].transform( df_train[ num_vars+cat_vars ] ).columns.tolist()


reg = pd.DataFrame( best_model[-1].coef_, columns=features ).T
reg = reg.rename(columns={0:"beta"})
reg['ratio'] = np.exp(reg["beta"])
reg['beta_10'] = 10 * reg["beta"]
reg['exp_beta_10'] = np.exp(10 * reg["beta"])


reg.sort_values( by="beta", ascending=False, inplace=True )
reg.to_excel("betas_logistica.xlsx")

model_s = pd.Series( {"cat_vars":cat_vars,
                      "num_vars":num_vars,
                      "fit_vars": X_train.columns.tolist(),
                      "model":best_model,
                      "auc":{"test": auc_test, "oot":auc_oot}} )

model_s.to_pickle("best_model_auto.pkl")

model_s
