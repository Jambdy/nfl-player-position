# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:36:36 2018

@author: James

Run experiments to predict player position from stats
"""

import process_nfl_data as nfl
import numpy as np
import sklearn as sk
import pandas as pd
 
def sum_stat(combine_stats):   
    df_mean = nfl.summary_stats(df_train,'Pos')
    cam = combine_stats[combine_stats['Player'].str.contains('Cam Newton')]
    nfl.display_scatter_pairs(combine_stats,['Height','Weight', '40YD'],cam)
    return df_mean
    
def dt_grid(df_train):
    # Run decision tree, find best parameters
    DT = nfl.decision_tree(df_train,'Pos')  
    params = {"min_samples_leaf":range(1,40,1),"criterion":["gini","entropy"]} 
    df_cv = nfl.grid_search(df_train,DT,'Pos',params,save=True)
    nfl.display_parameter_curve(df_cv,'param_min_samples_leaf','mean_test_score',
                                'Min Samples Leaf','Mean Test Score','param_criterion')
    nfl.display_parameter_curve(df_cv,'param_min_samples_leaf','mean_train_score',
                                'Min Sample Leaf','Mean Train Score','param_criterion')

def svm_grid(df_train):
    # Run support vector machine, find best parameters
    SVM = nfl.svm(df_train,'Pos') 
    params = {"C":np.power(10.0,np.arange(-6,5,1)),"kernel":["linear","rbf"]}  
    df_cv=nfl.grid_search(df_train,SVM,'Pos',params,save=True)
    nfl.display_parameter_curve(df_cv,'param_C','mean_test_score',
                                'Penalty Param','Mean Test Score','param_kernel',True)
    nfl.display_parameter_curve(df_cv,'param_C','mean_train_score',
                                'Penalty Param','Mean Train Score','param_kernel',True)

def rf_grid(df_train):
    # Run random forest, find best parameters
    RF = nfl.rf(df_train,'Pos') 
    params = {"n_estimators":range(1,420,20),"criterion":["gini","entropy"]} 
    df_cv = nfl.grid_search(df_train,RF,'Pos',params,save=True)
    nfl.display_parameter_curve(df_cv,'param_n_estimators','mean_test_score',
                                '# Estimators','Mean Test Score','param_criterion')

def rf_grid_est(df_train):
    # Run grid search multiple times for RT n_estimators
    RF = nfl.rf(df_train,'Pos') 
    params = {"n_estimators":range(1,420,20),"criterion":["gini","entropy"]} 
    
    for i in range(0,1):
        print(i)
        try:
            df_cv = df_cv.append(nfl.grid_search(df_train,RF,'Pos',params,save=False))
        except:
            df_cv = nfl.grid_search(df_train,RF,'Pos',params,save=False)

    df_cv = df_cv.groupby(['param_n_estimators']).mean().reset_index()
        
    nfl.display_parameter_curve(df_cv,'param_n_estimators','mean_test_score',
                                '# Estimators','Mean Test Score','param_criterion')
    nfl.display_parameter_curve(df_cv,'param_n_estimators','mean_train_score',
                                '# Estimators','Mean Train Score','param_criterion')

def rf_grid_maxfeature(df_train):
    # Run grid search multiple times for max features
    RF = nfl.rf(df_train,'Pos') 
    params = {"n_estimators":[220],"criterion":["gini"],"max_features":range(1,df_train.shape[1]-1,1)} 
    
    for i in range(0,10):
        print(i)
        try:
            df_cv = df_cv.append(nfl.grid_search(df_train,RF,'Pos',params,save=False))
        except:
            df_cv = nfl.grid_search(df_train,RF,'Pos',params,save=False)
    df_cv = df_cv.groupby(['param_max_features']).mean().reset_index()
            
    nfl.display_parameter_curve(df_cv,'param_max_features','mean_test_score',
                                'n_estimators','Mean Test Score')
    nfl.display_parameter_curve(df_cv,'param_max_features','mean_train_score',
                                'n_estimators','Mean Train Score')

def svm_test_set(df_train,df_test,df):
    SVM = sk.svm.SVC(C=10,kernel="linear")
    SVM.fit(df_train.iloc[:,1:],df_train.iloc[:,0])
    y_test = df_test.iloc[:,0]
    y_pred = SVM.predict(df_test.iloc[:,1:])
    print('Test accuracy: '+str(sk.metrics.accuracy_score(y_pred,y_test)))
    
    # Create confusion matrices
    labels = sorted(df['Pos'].unique())
    cm = sk.metrics.confusion_matrix(y_test,y_pred,labels=labels)
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:,None],decimals=2)
    nfl.save_confusion_matrix(cm,labels,'cm')
    nfl.save_confusion_matrix(cm_norm,labels,'cm_norm')
    
    nfl.confusion_matrix_heatmap(cm_norm,labels)
    return cm_norm
           
def svm_test_set_prob(df_train, df, cam):
    SVM = sk.svm.SVC(C=10,kernel="linear",probability=True)
    SVM.fit(df_train.iloc[:,1:],df_train.iloc[:,0])
    
    # Show Cam Newton's stats
    prob_cam = SVM.predict_proba(cam.iloc[:,1:])

    df_prob = pd.DataFrame({'Pos':sorted(df['Pos'].unique()),'Prob':prob_cam[0,:]})
    nfl.prob_pie_chart(df_prob)
    
def svm_test_samples_vs_acc(cm):
    df_mean,samples = nfl.summary_stats(df_train,'Pos')
    df_plt = pd.DataFrame({'Sample Size':samples,'Classification Accuracy':cm.diagonal()})
    nfl.display_scatter(df_plt,'Sample Size','Classification Accuracy')


combine_stats = nfl.load_nflcombine_data()
df = combine_stats[['Pos','Height','Weight', '40YD', 'Vertical', 'BenchReps', 'Broad Jump','3Cone', 'Shuttle']]
df = nfl.handle_missing_and_cat_data(df)
#cam = df[combine_stats['Player'].str.contains('Cam Newton')]
cam = pd.DataFrame({'Pos':'RB','Height':60,'Weight':135, '40YD':6, 'Vertical':12, 'BenchReps':0, 'Broad Jump':40,'3Cone':14, 'Shuttle':10},index=[0])
df_train, df_test = nfl.split_test_training_sets(df)  

#cm = svm_test_set()
#df_mean = sum_stat()
#svm_test_samples_vs_acc(cm)

svm_test_set_prob(df_train, df, cam)
