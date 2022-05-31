import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd
import numpy as np

""" Permet d'afficher les mètres carrés en fonction du prix """
def plot_squareMeters(df):
    plt.scatter(df[['squareMeters']], df[['price']], alpha = 0.5)
    plt.xlabel('squareMeters')
    plt.ylabel('price')
    plt.show()

""" Compare les valeurs des différentes données ordonnées par feature """
def plot_features(df):
    df.hist(bins = 50, figsize = (20,20))
    plt.show()

""" Affiche les informations de la dataframe """
def print_df_info(df):
    print("\nDF.head():")
    print(df.head())
    print("\nDF.info():")
    print(df.info())
    print("\nDF.describe():")
    print(df.describe())

""" Récupère X et y depuis la dataframe """
def get_Xy(df):
    X = df.reindex(columns = df.columns[:-1]);  X = X.values.tolist();
    y = df.reindex(columns = [df.columns[-1]]); y = y.values.tolist();
    return X,y

def get_dict_features(X, feature_names):
    dict_features = {}
    dict_features['squareMeters']       = [[Xi[0] for Xi in X]]
    dict_features['numberOfRooms']      = [[Xi[1] for Xi in X]]
    dict_features['hasYard']            = [[Xi[2] for Xi in X]]
    dict_features['hasPool']            = [[Xi[3] for Xi in X]]
    dict_features['floors']             = [[Xi[4] for Xi in X]]
    dict_features['cityCode']           = [[Xi[5] for Xi in X]]
    dict_features['cityPartRange']      = [[Xi[6] for Xi in X]]
    dict_features['numPrevOwners']      = [[Xi[7] for Xi in X]]
    dict_features['made']               = [[Xi[8] for Xi in X]]
    dict_features['isNewBuilt']         = [[Xi[9] for Xi in X]]
    dict_features['hasStormProtector']  = [[Xi[10] for Xi in X]]
    dict_features['basement']           = [[Xi[11] for Xi in X]]
    dict_features['attic']              = [[Xi[12] for Xi in X]]
    dict_features['garage']              = [[Xi[13] for Xi in X]]
    dict_features['hasStorageRoom']     = [[Xi[14] for Xi in X]]
    dict_features['hasGuestRoom']       = [[Xi[15] for Xi in X]]
    return dict_features

""" applique la regression """ 
def get_regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    return reg, X_train, X_test, y_train, y_test

""" applique la prédiction """
def prediction(reg, X_test, y_test):
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

""" affiche les résultats de la prédiction """
def print_res(y_test, y_pred):
    intercept, slope, corr_coeff = plot_linear_regression(y_test, y_pred)
    plt.show()

""" Lance le test l'élimination de features récursif """

def test_rfe(train_set, label_train, lin_reg):
    res = []
    for i in range(1,17):
        rfe_i = RFE(lin_reg, n_features_to_select=i, step=1)
        rfe_i = rfe_i.fit(train_set, label_train)
        predictions_rfe_i = rfe_i.predict(train_set)
        lin_mse_rfe_i = mean_squared_error(label_train, predictions_rfe_i)
        lin_rmse_rfe_i = np.sqrt(lin_mse_rfe_i)
        print(i," ",lin_rmse_rfe_i)
        res.append(lin_rmse_rfe_i)

    plt.plot(range(1,17),res)
    plt.ylim([0,7000])
    plt.show()

""" Renvoie les meilleurs features choisies par SelectKBest() """

def get_best_features(X_train, y_train, X_test, feature_names, n_feature):
    fs = SelectKBest(score_func=f_regression, k='all')
    fs.fit(X_train, y_train)
    for i in range(len(fs.scores_)):
        print(f'{feature_names[i]}: {fs.scores_[i]}')

    # plot the scores
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.ylim([0,10])
    plt.show()

    coef_copy = fs.scores_.copy()
    coef_copy.sort()
    best_f = []
    coef_copy = coef_copy[::-1]
    for i in range(n_feature):
        best_f.append(np.where(fs.scores_==coef_copy[i]))	

    best_features = [feature_names[i[0][0]] for i in best_f]

    return best_features

""" Filtre les features """

def filter_features(feature_names, best_features, df):
    not_chosen_features = [x for x in feature_names if not x in best_features]
    print("features_names again and again: ",feature_names)
    print(f"\n\tSupprimons les features:\n{not_chosen_features}")

    for feature in not_chosen_features:
        df.pop(feature)

    feature_names, _ = get_feature_target_names(df)	
    X, y = get_Xy(df)
    print(f"dimension de X: {len(X[0])}")
    return X,y,feature_names

""" Lance le test ou la moins bonne des features est retirée jusqu'à ce qu'il n'en reste qu'une """

def plot_with_best_features(features_start, df, n_feature_min=1):
    nb_feature = len(features_start)
    results = []
    X,y = get_Xy(df)
    print(f"features: {features_start}\nlen: {nb_feature}")
    print(range(nb_feature))
    for i in range(nb_feature):
        reg, X_train, X_test, y_train, y_test = get_regression(X,y)
        best_features = get_best_features(X_train, y_train, X_test, features_start, len(features_start)-1)
        X,y, features_start = filter_features(features_start, best_features, df)
        result = prediction(reg,X_test, y_test)
        results.append(result)
        print(f"\nFor {len(features_start)} features, precision is {result}")
        if len(X[0]) < n_feature_min:
            break
    list_features_descending = list(reversed(range(1,nb_feature+1)))
    print(f"x:{list_features_descending}\nresults: {results}")
    plt.plot(list_features_descending, results)
    plt.show()

""" Renvoie le nom des features et le nom de la cible en assumant que 
    la dernière colonne est la cible """

def get_feature_target_names(dataframe, wanted_features = None):
    #	 split the last element of the dataframe. (-1 -> len()-1)
    columns = dataframe.columns
    if wanted_features == None:
        feature_names = list(columns[:-1])
    else:
        feature_names = list(set(columns).intersection(set(wanted_features)))

    target_name = columns[-1]

    return feature_names, target_name

""" Présente les différentes fonctionalités et plus montrées dans l'article """

def test_article():

    """ On récupère le jeu de données Xy depuis le fichier .csv """
    df = pd.read_csv("ParisHousing.csv")
    feature_names, target_name = get_feature_target_names(df)
    X, y = get_Xy(df)
    X = np.array(X)
    y = np.array(y)
        
    print("Voici nos features")
    plot_features(df)

    print("Voici les informations concernant notre base de donnée")
    print_df_info(df)

    print("Courbe de y = x * prix du mètre au carré")
    plot_squareMeters(df)

    """ Split de la data + entraînement du modèle	"""	
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"mean absolute error : {mae} average appartement price: {sum(y_pred)/len(y_pred)}\nmae in percent: {mae/(sum(y_pred)/len(y_pred))}")

    """ 
    Calcul des meilleurs features + affichage matplotlib """
    fs = SelectKBest(score_func=f_regression, k='all')

    fs.fit(X_train, y_train)
    for i in range(len(fs.scores_)):
        print(f'{feature_names[i]}: {fs.scores_[i]}')

    # plot the scores
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.ylim([0,10])
    plt.xlabel("features")
    plt.ylabel("coef")
    plt.show()
    """ Elimination des features moins efficaces et affichage resultat """
    # Prenons les 10 meilleurs features:
    to_remove = []
    for i in range(len(fs.scores_)):
        if fs.scores_[i] < 0.30:
            to_remove.append(feature_names[i])
    filtered_features = [feature for feature in feature_names if not feature in to_remove]
    X = df.reindex(columns =filtered_features)
    X = X.values.tolist()

    print(f"nouvelle dimension de X: {len(X[0])}")
    print(f"Features choisies:\n{filtered_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"resultat avec réduction de features: {mae}")

""" L'interpreteur commencera ici """

if __name__ == "__main__":

    df = pd.read_csv("ParisHousing.csv")
    feature_names, target_name = get_feature_target_names(df)

    test_article()
    



