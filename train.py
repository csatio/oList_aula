import datetime
import feature_engine.missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt
import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import tree     # Árvore de Decisão
from sklearn import ensemble # Random Forest
import xgboost as xgb        # XGBoost

import sqlalchemy

def train_rl(imputer_zero,imputer_um,onehot,X_train,y_train):
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

    return search.best_estimator_

    #best_model_rl = search.best_estimator_

def train_tree(imputer_zero,imputer_um,onehot,X_train,y_train):
    model_tree = tree.DecisionTreeClassifier() # Definição do modelo

    full_pipeline_tree = Pipeline( steps=[ ('zero', imputer_zero),
                                        ('um', imputer_um),
                                        ("onehot", onehot),
                                        ('modelo', model_tree) ] )

    param_grid = { "modelo__max_depth":[None, 3,4,5,6,7,8,9,10],
                "modelo__min_samples_split":[2,5,10],
                "modelo__min_samples_leaf":[1,2,3,4,5,7,10,20] }

    search_tree = model_selection.GridSearchCV(full_pipeline_tree,
                                            param_grid,
                                            cv=3,
                                            n_jobs=-1,
                                            scoring='neg_root_mean_squared_error') # Declaração

    search_tree.fit(X_train, y_train) # Executa o treinamento!!

    return search_tree.best_estimator_


def train_rf(imputer_zero,imputer_um,onehot,X_train,y_train):
    model_rf = ensemble.RandomForestClassifier(random_state=1992) # Definição do modelo

    full_pipeline_rf = Pipeline( steps=[('zero', imputer_zero),
                                        ('um', imputer_um),
                                        ('onehot', onehot),
                                        ('modelo', model_rf)])

    param_grid = { "modelo__n_estimators":[10,20,50,100],
                "modelo__max_depth":[5,10],
                "modelo__min_samples_split":[10,12],
                "modelo__min_samples_leaf":[5,10,50] }

    search_rf = model_selection.GridSearchCV(full_pipeline_rf,
                                            param_grid,
                                            cv=3,
                                            n_jobs=-1,
                                            scoring='neg_root_mean_squared_error') # Declaração

    search_rf.fit(X_train, y_train) # Executa o treinamento!!

    return search_rf.best_estimator_

def train_xgb(model_xgb, steps,param_grid,X_train,y_train):
    

    full_pipeline_xgb = Pipeline( steps=steps)

    param_grid = param_grid

    search_xgb = model_selection.GridSearchCV(full_pipeline_xgb,
                                            param_grid,
                                            cv=3,
                                            n_jobs=-1,
                                            scoring='neg_root_mean_squared_error') #Declaração

    search_xgb.fit(X_train, y_train) #Executa o treinamento!!

    return search_xgb.best_estimator_


def get_models(DATA_PATH,model_path):
    '''function to train all models and choose best one
    '''
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

    # Regressão Logística

    best_model_rl = train_rl(imputer_zero,imputer_um,onehot,X_train,y_train)


    # Arvore decisão
    
    best_model_tree = train_tree(imputer_zero,imputer_um,onehot,X_train,y_train)


    #Random forest
    

    best_model_rf = train_rf(imputer_zero,imputer_um,onehot,X_train,y_train)


    #XGBoost 1
    model_xgb = xgb.XGBClassifier(random_state=1992)

    steps=[('zero', imputer_zero),
                                        ('um', imputer_um),
                                        ('onehot', onehot),
                                        ('modelo', model_xgb)]

    param_grid = { "modelo__n_estimators":[90],
                "modelo__max_depth":[4],
                "modelo__eta":[0.1],
                "modelo__subsample":[0.9] }

    #param_grid = { "modelo__n_estimators":[90,100,110],
    #            "modelo__max_depth":[4,5,6],
    #            "modelo__eta":[0.05,0.1, 0.15],
    #            "modelo__subsample":[0.85, 0.9,0.95] }

    best_model_xgb= train_xgb(model_xgb,steps,param_grid,X_train,y_train)


    #XGBOOST 2

    model_xgb = xgb.XGBClassifier(random_state=1992)

    full_pipeline_xgb_nt = Pipeline( steps=[('onehot', onehot),
                                            ('modelo', model_xgb)])

    param_grid = {"modelo__n_estimators":[10],
                "modelo__max_depth":[3],
                "modelo__eta":[0.1],
                "modelo__subsample":[0.1] }

    #param_grid = {"modelo__n_estimators":[10,50,100],
    #            "modelo__max_depth":[3,5,10],
    #            "modelo__eta":[0.1, 0.3, 0.7, 0.9],
    #            "modelo__subsample":[0.1, 0.2, 0.5, 0.8, 0.9] }

    best_model_xgb_nt = train_xgb(model_xgb,steps,param_grid,X_train,y_train)


    # Verificando erro na base de teste
    y_test_prob_rl = best_model_rl.predict_proba(X_test)[:, 1]
    y_test_prob_tree = best_model_tree.predict_proba(X_test)[:, 1]
    y_test_prob_rf = best_model_rf.predict_proba(X_test)[:, 1]
    y_test_prob_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
    y_test_prob_xgb_nt = best_model_xgb_nt.predict_proba(X_test)[:, 1]

    auc_test_rl = metrics.roc_auc_score( y_test, y_test_prob_rl)
    auc_test_tree = metrics.roc_auc_score( y_test, y_test_prob_tree)
    auc_test_rf = metrics.roc_auc_score( y_test, y_test_prob_rf)
    auc_test_xgb = metrics.roc_auc_score( y_test, y_test_prob_xgb)
    auc_test_xgb_nt = metrics.roc_auc_score( y_test, y_test_prob_xgb_nt)


    print( "ROC Regressão Logística", auc_test_rl)
    print( "ROC Árvore", auc_test_tree)
    print( "ROC Random Forest", auc_test_rf)
    print( "ROC XGB", auc_test_xgb)
    print( "ROC XGB Sem tratamento", auc_test_xgb_nt)

    # add results to pandas dataframe

    all_models = pd.DataFrame(columns=['name','modelo','auc'],
                              data=[['Regressao Logistica',best_model_rl,auc_test_rl],
                              ['Arvore',best_model_tree,auc_test_tree],
                              ['Forest',best_model_rf,auc_test_rf],
                              ['XGB1',best_model_xgb,auc_test_xgb],
                              ['XGB2',best_model_xgb_nt,auc_test_xgb_nt],
                              ])

    # sort by auc result
    campeao_model = all_models.sort_values('auc',ascending=False).head(1)['modelo'].item()
    campeao_auc = all_models.sort_values('auc',ascending=False).head(1)['auc'].item()

    campeao_nome = all_models.sort_values('auc',ascending=False).head(1)['name'].item()


    campeao_model.fit( df_train[num_vars+cat_vars], df_train[target] )

    # Verificando erro na base de Out of Time
    y_test_prob = campeao_model.predict_proba(df_oot[ num_vars+cat_vars ])[:, 1]
    auc_oot = metrics.roc_auc_score( df_oot[target], y_test_prob)
    print( "Área sob a curva ROC:", auc_oot)


    campeao_model.fit(df[num_vars+cat_vars], df[target])

    features = campeao_model[:-1].transform( df_train[ num_vars+cat_vars ] ).columns.tolist()

    try:

        features_importance = pd.Series(campeao_model[-1].feature_importances_, index=features)

    except AttributeError:
        features_importance = ''


    features_importance.sort_values(ascending=False).head(20)

    

    model_s = pd.Series( {"cat_vars":cat_vars,
                        "num_vars":num_vars,
                        "fit_vars": X_train.columns.tolist(),
                        "model":campeao_model,
                        "auc":{"test": campeao_auc, "oot":auc_oot}} )

    now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    model_s.to_pickle(model_path + f'champion_model_{now}.pkl')

    resultado = [campeao_nome,campeao_auc]

    return resultado

