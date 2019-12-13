import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 100)

import numpy as np

import pandas_profiling

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("dark")
sns.set_context("talk")
import itertools

from statistics import mode

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus



def delete_entries(df, column, values):
    """
    Takes in a df, column name (string) and list of values to be deleted from the column
    
    """
    for val in values:
        dropindex = df[df[column] == val].index
        df.drop(index = dropindex, inplace = True)
        
        

def plot_cf_target(df, columns, hue = "default_payment_next_month"):
    """
    Takes in a df, list of cf column headers then returns pairwise plots
    
    """
    
    loc = 1
    numplots = ((len(columns)*(len(columns)-1))/2)+len(columns)
    fig = plt.figure(figsize = (40,((numplots/4)+1)*8))
    for i, ycol in enumerate(columns):
        for j, xcol in enumerate(columns):
            if j < i:
                continue
            else:
                ax = fig.add_subplot((numplots/4)+1, 4, loc)
                if xcol == ycol:
                    sns.distplot(df[xcol], ax = ax);
                else:
                    sns.scatterplot(x=xcol, y=ycol, data=df, palette = "GnBu_d", ax = ax, hue = hue, legend = False);
            loc += 1

            
def plot_cat(df, cat_columns, hue = "default_payment_next_month"):
    """
    Takes in a df, list of cat column headers then returns bar plots vs # of clients
    
    """
    fig = plt.figure(figsize = (20,(len(cat_columns)/2+1)*8))
    loc = 1
    for col in cat_columns:
        ax = fig.add_subplot(len(cat_columns)/2+1, 2, loc)
        if (col == "score_3mo") | (col == "score_6mo"):
            df_plot = df[[col, hue, "id"]].groupby([col, hue]).count()
            df_plot.reset_index(inplace = True)
            sns.lineplot(x=col, y="id",hue=hue, data=df_plot, palette = "GnBu_d", ax = ax);
            plt.ylim([0.0001,1200])
            plt.ylabel("clients");
        else:
            df_plot = df[[col, hue, "id"]].groupby([col, hue]).count()
            df_plot.reset_index(inplace = True)
            sns.barplot(x=col, y= "id", hue = hue, data=df_plot, palette = "GnBu_d", ax = ax);
            plt.legend(title = "default payment (1=yes, 0=no)")
            plt.ylim([0.0001,3000])
            plt.ylabel("clients");
        loc += 1
        
        
def preprocess_data(df, cat_columns, cf_columns):
    """ 
    This function takes in dataframe df, listst cat_columns, cf_columns and preprocess the data 
    
    """
    
    #define cat and cf columns, convert cat to dummies
    df_cat = pd.get_dummies(df[cat_columns], columns = cat_columns, drop_first=True, dtype=float)
    df_cat.rename(mapper= {"sex_2": "sex_female", "education_2":"education_university", "education_3": "education_high_school", "education_4":"education_others","marital_status_2":"marital_status_single", "marital_status_3": "marital_status_others"}, axis = 1, inplace = True)
    df_cf = df[cf_columns]
    X = pd.concat([df_cf, df_cat], axis = 1)
    y = df[['default_payment_next_month']]
    print("dummy variables created")
    
    #train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print("split done")
    
    #resample the train sets
    smote = SMOTE(sampling_strategy = "not majority", random_state = 42)
    X_train_rs, y_train_rs = smote.fit_sample(X_train, y_train)
    print('original class distribution:')
    print(y["default_payment_next_month"].value_counts())
    print('synthetic sample class distribution:')
    print(pd.Series(y_train_rs).value_counts())  
    return X, X_train_rs, X_test, y_train_rs, y_test

def scale_X(X_train, X_test):
    """ 
    This function takes in X_train, X_test arrays then return scaled X_train, X_test arrays
    
    """
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("scaling done")
    
    return X_train_scaled, X_test_scaled


def feature_selection_Logistic(df, cat_columns, cf_columns):
    """ 
    This function takes in df, list of cat and cf column headers then prepocess, split, resampled and scaled the data.
    Then uses SelectFromModel to select features. 
    
    """
    
    X, X_train_rs, X_test, y_train_rs, y_test = preprocess_data(df, cat_columns, cf_columns)
    X_train_scaled, X_test_scaled = scale_X(X_train_rs, X_test)
    
    #run SelectFromModel for feature selection
    selector = SelectFromModel(LogisticRegression(fit_intercept=True, C=1e20, penalty ='l2', solver = 'lbfgs'))
    selector.fit(X_train_scaled, y_train_rs)
    selected_feat = list(X.columns[(selector.get_support())])
    selected = dict(zip((list(X.columns)), list(selector.get_support())))
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    print(sum(selector.get_support())," features selected out of ", len(list(X.columns)))
    
    return selected, X_train_selected, X_test_selected, y_train_rs, y_test


def run_logistic(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(fit_intercept=True, C=1e20, penalty ='l2', solver = 'lbfgs')
    logreg.fit(X_train, y_train)
    get_scores(logreg, X_train, X_test, y_train, y_test)


def run_svc(X_train, X_test, y_train, y_test):
    svc = SVC(max_iter=100000,probability=True)
    svc.fit(X_train, y_train)
    get_scores(svc, X_train, X_test, y_train, y_test)

    

def get_scores(estimator, X_train, X_test, y_train, y_test): 
    y_test_pred = estimator.predict(X_test)
    probas = estimator.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, probas[:,1])
#     print("Precision: ", precision)
#     print("Recall: ", recall)
#     print("Precision-Recall Curve", precision_recall_curve(y_test, probas[:,1]))
    print("---------------------------------------------------------------------")
    print("F1 score: ", f1_score(y_test, y_test_pred))
    plot_auc(recall, precision, thresholds)
    print("---------------------------------------------------------------------")
    print("Classification Report: \n", classification_report(y_test, y_test_pred))
    print("---------------------------------------------------------------------")
    plot_confusion_matrix(confusion_matrix(y_test, y_test_pred), classes=[0, 1])
    
    
def plot_auc(recall, precision, thresholds):
    sns.set_style("darkgrid")
    sns.set_context("paper")
    print('Precision-Recall AUC: {}'.format(auc(recall, precision)))
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='Precision-Recall Curve')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.00001, 1.00001])
    plt.ylim([0.00001, 1.00001])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize = 15)
    plt.legend(loc='lower right')
    plt.show()
    print("---------------------------------------------------------------------")
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], "b--", label="Precision", lw =2)
    plt.plot(thresholds, recall[:-1], "g-", label="Recall", lw =2)
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.title('Precision-Recall vs Decision Threshold', fontsize = 15)
    plt.show()
    
#     plt.figure(figsize=(8, 6))
#     lw = 2
#     plt.plot(recall, precision, color='darkorange',
#              lw=lw, label='Precision-Recall Curve')
#     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.00001, 1.00001])
#     plt.ylim([0.00001, 1.00001])
#     plt.yticks([i/20.0 for i in range(21)])
#     plt.xticks([i/20.0 for i in range(21)])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve', fontsize = 15)
#     plt.legend(loc='lower right')
#     plt.show()
    

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    sns.set_style("darkgrid")
    sns.set_context("paper")
    #Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 verticalalignment='top',
                 color='white' if cm[i, j] > thresh else 'black',
                 fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    
    

def run_GridSearchCV(X_train, X_test, y_train, y_test, model, param_grid, scoring, cv = 10):
    opt_model = GridSearchCV(model, param_grid = param_grid, scoring = scoring, cv = cv, n_jobs = -1, return_train_score = True)
    print("running gridsearch...")
    opt_model.fit(X_train, y_train)
    best_model = opt_model.best_estimator_
    print("done!")
    print("---------------------------------------------------------------------")
    print("Best Parameters:", opt_model.best_params_)
    
    
    get_scores(best_model, X_train, X_test, y_train, y_test)

    cv_results = pd.DataFrame(opt_model.cv_results_)
    columns = ["rank_test_score", "params", "mean_train_score","std_train_score", "mean_test_score", "std_test_score"]
    cv_results_top10 = cv_results[columns].sort_values("rank_test_score").head(10)
    
    return cv_results_top10, best_model


def split_resample(X, y):
    """ 
    This function takes in dataframes X, y the preprocess the data: split then resample
    
    """
    
    #train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print("split done")
    
    #resample the train sets
    smote = SMOTE(sampling_strategy = "not majority", random_state = 42)
    X_train_rs, y_train_rs = smote.fit_sample(X_train, y_train)
    print('original class distribution:')
    print(y["default_payment_next_month"].value_counts())
    print('synthetic sample class distribution:')
    print(pd.Series(y_train_rs).value_counts())  
    return X, X_train_rs, X_test, y_train_rs, y_test
