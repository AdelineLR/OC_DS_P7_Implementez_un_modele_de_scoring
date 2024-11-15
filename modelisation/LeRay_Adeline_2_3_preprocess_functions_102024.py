# Author : Le Ray Adeline
# OpenClassrooms Parcours Data Scientist
# Projet : P7 Implémentez un modèle de scoring
# Date : October 2024
# Content : Data preprocessing functions


import numpy as np
import pandas as pd 
import missingno as msno 
from skimpy import skim
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from contextlib import contextmanager
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


"""
Loading and inspection function
"""
# Extract columns descriptions for defined table from HomeCredit_columns_description.csv and display result
def show_col_description(table_name):
    # Set options to display all rows and columns without truncation
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_colwidth', None)  # Show full column content without truncation
    
    # HomeCredit_columns_description
    HomeCredit_columns_description = pd.read_csv('input/HomeCredit_columns_description.csv', sep=',', encoding='ISO-8859-1')
    # Drop the 'Unnamed: 0' column
    HomeCredit_columns_description = HomeCredit_columns_description.drop(columns=['Unnamed: 0'])

    # Filter the DataFrame and display specific columns
    display(HomeCredit_columns_description.loc[HomeCredit_columns_description['Table'] == table_name, ['Row', 'Description']])

    # Reset options to default values after display
    pd.reset_option('display.max_rows')  # Reset to default row display limit
    pd.reset_option('display.max_colwidth')  # Reset to default column width
    
# Read table and display head
def load_table(table_name, num_rows=None):
    # Read dataset
    df = pd.read_csv('input/'+table_name, nrows = num_rows)
    print(f'{table_name} data shape: ', df.shape)
    
    return df

# Fonction pour inspection df
def inspection(df, missing_graph=False):
    """!
    @brief Effectue une inspection approfondie du dataframe.

    Cette fonction affiche les premières lignes, les dimensions, les types de variables, 
    les valeurs manquantes, supprime les colonnes vides et avec une seule modalité, identifie les doublons,
    présente une description statistique du dataframe et un échantillon des modalités des variables qualitatives.

    @param df: Dataframe à inspecter (pandas DataFrame).
    """
    
    # Options d'affichage : toutes les lignes
    pd.set_option('display.max_rows', None)
      
    # Affichage des premières lignes du dataframe
    print("Dataframe")
    print("-" * 50)
    display(df.head())
    print("")
    
    # Supression des colonnes vides ou avec une unique modalité et sans valeurs manquantes
    del_col(df)

    # Affichage des dimensions, types de variables et valeurs non-null, 
    # Description statistique du dataframe (moyenne, écart-type, min-max, médiane, IQR)
    print("Dimensions du dataframe, Types de variables, Description statistique du dataframe, Valeurs non-null")
    print("-" * 50)
    print(skim(df))
    print("")
          
    # Affichage des valeurs uniques par colonne
    print("Valeurs uniques par variable")
    print("-" * 50)   
    print(df.nunique())
    print("")
    
    # Identification et affichage des doublons
    print("Nombre de doublons")
    print("-" * 50)   
    print(df.duplicated().sum())
    print("")
    
    # Affichage d'un échantillon des modalités des variables qualitatives (si applicable)
    col = df.select_dtypes(include='object').columns.tolist()
    
    if len(col)>0:
        print("")
        print("Echantillon des modalités des variables qualitatives (5 modalités max)")
        print("-" * 50) 
        for c in col:
            print(f'{c} : {df[c].unique()[:5]}\n')   
            
    # Affichage du graphique des valeurs manquantes (si applicable)
    if sum(df.isna().sum()) > 0 and missing_graph:
        print("")
        print("Graphique des valeurs manquantes")
        print("-" * 50)    
        msno.bar(df)
        msno.matrix(df)
  
    # Réinitialiser l'option pour revenir aux paramètres par défaut
    pd.reset_option('display.max_rows')
    
    
def del_col(df):
    """!
    @brief Supprime les colonnes du DataFrame selon certains critères.

    Cette fonction supprime les colonnes vides (contenant uniquement des valeurs NaN), ainsi que les colonnes 
    qui ne contiennent aucune valeur manquante et qui n'ont qu'une seule modalité.

    @param df : Le DataFrame à nettoyer. (type: pandas.DataFrame)
    """    
    mean_na = df.isna().mean()
    nunique = df.nunique()

    # Suppression des colonnes vides le cas échéant
    if max(mean_na) == 1:
        print("")
        print("Suppression des colonnes vides")
        print("-" * 50) 
        print('Dimensions du dataframe avant suppression :', df.shape)
        col_null = list(mean_na[mean_na == 1].index)
        print('Colonne(s) supprimée(s) :', col_null)
        df.drop(columns=col_null, inplace=True)
        print('Dimensions du dataframe après suppression :', df.shape)
        print("")
 
    # Suppression des colonnes sans valeurs manquantes et avec une seule modalité
    onemodal = pd.DataFrame({'mean_na' : mean_na,
                             'nunique' : nunique})
    onemodal = onemodal[(onemodal['mean_na'] == 0) & (onemodal['nunique'] == 1)]
    
    if onemodal.shape[0] >0:
        print("")
        print("Suppression des colonnes sans valeurs manquantes et avec une seule modalité")
        print("-" * 50) 
        print('Dimensions du dataframe avant suppression :', df.shape)
              
        onemodal = onemodal[(onemodal['mean_na'] == 0) & (onemodal['nunique'] == 1)]
        col_to_drop = list(onemodal.index)
        
        print('Colonne(s) supprimée(s) et modalités:')
        for c in col_to_drop:
            print(f'{c} = {df[c].unique()}')
        
        df.drop(columns=col_to_drop, inplace=True)
        print('Dimensions du dataframe après suppression :', df.shape)
        print("")
        

        

"""
Preprocessing functions
"""

def encode_categorical_columns(df):
    """!
    @brief Encodes categorical columns with 2 or fewer unique values using Label Encoding
    and applies One-Hot Encoding for the rest of the categorical columns.
    
    @param df (pd.DataFrame): The input DataFrame to encode.
    @returns : df (pd.DataFrame), le_col (list), encoded_col (list): The encoded DataFrame, the label encoded columns, and the final list of encoded columns.
    """
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    le_col = []
    one_hot_encoded_col = []

    # Iterate through the columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories, apply Label Encoding
            if len(list(df[col].unique())) <= 2:
                # Fit_transform data
                df[col] = le.fit_transform(df[col])
                
                # Keep track of how many columns were label encoded and their names
                le_count += 1
                le_col.append(col)
            else:
                # Track columns that will be one-hot encoded
                one_hot_encoded_col.append(col)
    
    # Save original column names before one-hot encoding
    original_cols = df.columns.tolist()

    # One-hot encoding of remaining categorical variables
    df = pd.get_dummies(df)

    # Get new column names after one-hot encoding
    new_encoded_cols = [col for col in df.columns if col not in original_cols]

    # Combine label encoded and one-hot encoded columns
    encoded_cols = le_col + new_encoded_cols

    # Return the DataFrame and the final list of encoded columns
    return df, encoded_cols



"""
 
"""
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(bureau_name, bb_name, num_rows=None):
    # Load     
    print(f"\n{'*' * 50}")
    print(f"Dataframe {bureau_name}")
    print(f"{'*' * 50}\n")
    bureau = load_table(bureau_name, num_rows)
    # Show columns description and Data inspection
    show_col_description(bureau_name)
    inspection(bureau) 
    
    # Load 
    print(f"\n{'*' * 50}")
    print(f"Dataframe {bb_name}")
    print(f"{'*' * 50}\n")
    bb = load_table(bb_name, num_rows)
    # Show columns description and Data inspection
    show_col_description(bb_name)
    inspection(bb)  
    
    # Encoding for categorical columns
    bureau, bureau_cat = encode_categorical_columns(bureau)
    bb, bb_cat = encode_categorical_columns(bb)
        
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = pd.merge(bureau, bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    import gc
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    
    # Bureau and bureau_balance categorical features
    cat_aggregations = {cat: ['mean'] for cat in bureau_cat}
    cat_aggregations.update({cat + "_MEAN": ['mean'] for cat in bb_cat})
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.merge(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    import gc

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.merge(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    import gc
    
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(prev_name, num_rows=None):
    # Load     
    print(f"\n{'*' * 50}")
    print(f"Dataframe {prev_name}")
    print(f"{'*' * 50}\n")
    prev = load_table(prev_name, num_rows)
    
    # Show columns description and Data inspection
    show_col_description(prev_name)
    inspection(prev)
    
    # Encoding for categorical columns  
    prev, cat_cols = encode_categorical_columns(prev)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.merge(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.merge(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(pos_name, num_rows=None):
    # Load 
    print(f"\n{'*' * 50}")
    print(f"Dataframe {pos_name}")
    print(f"{'*' * 50}\n")
    pos = load_table(pos_name, num_rows)
    
    # Show columns description and Data inspection
    show_col_description(pos_name)
    inspection(pos) 

    # Encoding for categorical columns
    pos, cat_cols = encode_categorical_columns(pos)
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(ins_name, num_rows=None):
    # Load 
    print(f"\n{'*' * 50}")
    print(f"Dataframe {ins_name}")
    print(f"{'*' * 50}\n")
    ins = load_table(ins_name, num_rows)
    
    # Show columns description and Data inspection
    show_col_description(ins_name)
    inspection(ins)
    
    # Encoding for categorical columns
    ins, cat_cols = encode_categorical_columns(ins)
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(cc_name, num_rows=None):
    # Load
    print(f"\n{'*' * 50}")
    print(f"Dataframe {cc_name}")
    print(f"{'*' * 50}\n")
    cc = load_table(cc_name, num_rows)
    
    # Show columns description and Data inspection
    show_col_description(cc_name)
    inspection(cc)
    
    # Encoding for categorical columns
    cc, ccb_cat = encode_categorical_columns(cc)
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def aggregate_tables(df,debug = False):
    num_rows = 10000 if debug else None
    if debug: 
        df = df.iloc[:num_rows, :]
    
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance('bureau.csv', 'bureau_balance.csv', num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications('previous_application.csv', num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash('POS_CASH_balance.csv', num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments('installments_payments.csv', num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance('credit_card_balance.csv', num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
        
    return df


# Conserver uniquement les colonnes avec moins d'un certain pourcentage de valeurs manquantes
def filter_missing_val(df, threshold):
    missing_values_percentage = df.isna().mean() * 100
    columns_to_keep = missing_values_percentage[missing_values_percentage < threshold].index
    df_filtered = df[columns_to_keep]
    return df_filtered
"""
Feature selection
"""
def custom_cost_function(y_true, y_pred):
    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Define the weights (custom cost)
    tp_weight = 0
    fp_weight = 1
    tn_weight = 0
    fn_weight = 10
    
    # Calculate cost function
    cost = ((fp * fp_weight) + (fn * fn_weight)) / (tp + tn + fp + fn)
    
    return cost


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    if debug:
        df = df.iloc[:10000,:]
    
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            verbosity=-1
        )

        # Fit the model with early stopping
        clf.fit(train_x, train_y, 
                eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric='auc'
               )

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        # Store feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print(f'Fold {n_fold + 1} AUC : {roc_auc_score(valid_y, oof_preds[valid_idx]):.6f}')
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print(f'Full AUC score: {roc_auc_score(train_df["TARGET"], oof_preds):.6f}')
        
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    best_features = (feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:40])
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

    
    
""" 
Impute and split
"""
def impute_split(X, y, test_size=0.2, random_state=42, strategy="mean"):
    """!
    @brief : Impute missing values, split data into train and test sets.
    @param X: DataFrame or ndarray, feature matrix.
    @param y: Series or ndarray, target vector.
    @param test_size: float, proportion of data to include in the test set.
    @param random_state: int.
    @param strategy: str, imputation strategy ("mean", "median", "most_frequent", or "constant").
    @return X_train, X_test, y_train, y_test: scaled and split datasets with column names.
    """
    # Step 1: Impute missing values
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)

    # Convert the imputed data back to a DataFrame with original column names
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

    # Step 2: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed_df, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


