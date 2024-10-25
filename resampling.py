import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

def res_smote(data_imbalanced):
    smote = SMOTE(random_state=42)
    data_smote_X, data_smote_y = smote.fit_resample(data_imbalanced.iloc[:, :-1], data_imbalanced.Label)
    data_smote = pd.concat([data_smote_X, data_smote_y], axis=1)
    return data_smote

def res_svmsmote(data_imbalanced):
    svmsmote = SVMSMOTE(random_state=14)
    data_svmsmote_X, data_svmsmote_y = svmsmote.fit_resample(data_imbalanced.iloc[:, :-1], data_imbalanced.Label)
    data_svmsmote = pd.concat([data_svmsmote_X, data_svmsmote_y], axis=1)
    return data_svmsmote

def res_enn(data_imbalanced):
    enn = EditedNearestNeighbours()
    data_enn_X, data_enn_y = enn.fit_resample(data_imbalanced.iloc[:, :-1], data_imbalanced.Label)
    data_enn = pd.concat([data_enn_X, data_enn_y], axis=1)
    return data_enn

def res_tomek(data_imbalanced):
    tomek = TomekLinks()
    data_tomek_X, data_tomek_y = tomek.fit_resample(data_imbalanced.iloc[:, :-1], data_imbalanced.Label)
    data_tomek = pd.concat([data_tomek_X, data_tomek_y], axis=1)
    return data_tomek

def res_smoteenn(data_imbalanced):
    smoteenn = SMOTEENN()
    data_smoteenn_X, data_smoteenn_y = smoteenn.fit_resample(data_imbalanced.iloc[:, :-1], data_imbalanced.Label)
    data_smoteenn = pd.concat([data_smoteenn_X, data_smoteenn_y], axis=1)
    return data_smoteenn

def res_smotetomek(data_imbalanced):
    smotetomek = SMOTETomek(random_state=14)
    data_smotetomek_X, data_smotetomek_y = smotetomek.fit_resample(data_imbalanced.iloc[:, :-1], data_imbalanced.Label)
    data_smotetomek = pd.concat([data_smotetomek_X, data_smotetomek_y], axis=1)
    return data_smotetomek