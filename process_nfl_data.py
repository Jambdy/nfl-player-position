# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 22:26:18 2018

@author: James

Predict Player Position from Stats
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def load_nflcombine_data():
    """
    # Load basic player statistics (souce: https://www.kaggle.com/kendallgillies/nflstatistics)
    basic_stats = pd.read_csv('input_data/nflstatistics/Basic_Stats.csv',sep= ',')  
    basic_stats = basic_stats[basic_stats['Age']<120]
    basic_stats = basic_stats[basic_stats['Position'].notnull()]
    """    
    
    # Load combine statistics (source: https://www.kaggle.com/kbanta11/nfl-combine)
    combine_stats = pd.DataFrame()
    for filename in os.listdir('input_data/nfl-combine/'): 
        temp_df = pd.read_csv('input_data/nfl-combine/'+filename,sep= ',')
        temp_df['Side'] = filename[4:len(filename)-4]
        combine_stats = pd.concat([combine_stats, temp_df])
    
    # Convert height feature to numeric
    combine_stats['Height']=combine_stats['Height'].str[0].astype('int64')*12 + combine_stats['Height'].str[2:].astype('int64')
    
    combine_stats = combine_stats.rename(index=str, columns={"Wt": "Weight"})
    
    return combine_stats
    
def split_test_training_sets(df,one_hot=False):  
    # Replace missing fields with column averages        
    df=df.fillna(df.mean())
    
    for col in df.columns[1:]:
        if df[col].dtype == "object":
            if one_hot: 
                # Convert categorical fields into multiple binary fields   
                temp_ar = pd.get_dummies(df[col])
                df[list(temp_ar)] = temp_ar
            else:
                # Drop categorical columns
                df = df.drop([col],axis=1)   
        
    # Spit the data into a training set and test set 
    return train_test_split(df,test_size=0.33,random_state=1)

def summary_stats(df,col):
    # Dispaly summary positions by position   
    grouped = df.groupby(col)
    print(grouped.aggregate(np.mean))
    print(grouped.size())

def display_histogram(df):
    # Show histograms    
    for col,typ in df.dtypes.iteritems():      
        if typ in ['float64','int64']:
            plt.figure()
            df[col].plot.hist(title=col)
            
def display_scatter(df,x,y):
    # Show scatter for inputted fields     
    plt.figure()
    plt.scatter(df[x],df[y],color='black')
    plt.title(x+" vs "+y)
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Show Cam Newton's stats
    cam = df[df['Player'].str.contains('Cam Newton')]
    if len(cam) != 0:
        plt.scatter(cam[x],cam[y],color='#0083C9',marker='^',s=500)

def display_scatter_pairs(df,features,player):
    df = df[features]
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
    for f1 in range(0,len(features)):
        for f2 in range(0,len(features)):
            if f1 != f2:            
                plt.sca(axes[f1, f2])
                plt.scatter(player[features[f2]],player[features[f1]],color='#0083C9',marker='^',s=150)
    plt.tight_layout()
    plt.savefig('scatter_matrix.png')

def decision_tree(df,label):
    # Split dataset into features and inputted label
    y = df[label]
    X = df.drop([label],axis=1)
        
    # Train Decison Tree on training data with cross validation
    DT = DecisionTreeClassifier() 
    scores = cross_validate(DT,X,y,return_train_score=True)
    print(scores)
    
    return DT
    
def svm(df,label):
    # Split dataset into features and inputted label
    y = df[label]
    X = df.drop([label],axis=1)    
    
    # Train Decison Tree on training data with cross validation
    SVM = SVC() 
    scores = cross_validate(SVM,X,y,return_train_score=True)
    print(scores)    
    
    return SVM
    
def grid_search(df,clf,label,params,save=True):
    # Split dataset into features and inputted label
    y = df[label]
    X = df.drop([label],axis=1)  
    
    clf_gs = GridSearchCV(clf, params, return_train_score=True)
    clf_gs.fit(X,y)
    print(clf_gs.best_score_)
    print(clf_gs.best_params_)
    
    if save:
        df_cv = pd.DataFrame.from_dict(clf_gs.cv_results_)
        df_cv.to_csv('output/grid_result_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.csv')
            
            
    
def main():     
    combine_stats = load_nflcombine_data()
    #display_histogram(combine_stats)
    #display_scatter(combine_stats,'Height','Weight')    
    #cam = combine_stats[combine_stats['Player'].str.contains('Cam Newton')]
    #display_scatter_pairs(combine_stats,['Height','Weight', '40YD'],cam)
    
    df = combine_stats[['Pos','Height','Weight', '40YD', 'Vertical', 'BenchReps', 'Broad Jump','3Cone', 'Shuttle']]
    df_train, df_test = split_test_training_sets(df)  
    
    #summary_stats(df_train,'Pos')
    
    # Run decision tree, find best parameters
    DT = decision_tree(df_train,'Pos')  
    params = {"min_samples_leaf":range(1,30,5),"criterion":["gini","entropy"]} 
    grid_search(df_train,DT,'Pos',params)
    
    # Run support vector machine, find best parameters
    SVM = svm(df_train,'Pos') 
    #params = {"degree":range(1,5,1),"kernel":["poly"]} 
    params = {"C":np.arange(0.1, 1.1, 0.1),"kernel":["rbf"]} 
    grid_search(df_train,SVM,'Pos',params)
    return
    
if __name__ == "__main__":
    main()