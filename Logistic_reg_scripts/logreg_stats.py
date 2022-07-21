import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut

def calc_mcfad(X, y, reg=LogisticRegression()):
    lr = reg.fit(X,y)
    y_pred_prob = lr.predict_proba(X)[:,1] #List of predicted probabilities of output being "1" or above y-cut
    null_probability = np.count_nonzero(y)/len(y) # overall probability of being "1" in dataset

    # Calculate log likelihood and null log likelihood
    null_log_likelihood = 0
    for i in y:
        if i == 1:
            null_log_likelihood += np.log(null_probability)
        elif i == 0:
            null_log_likelihood += np.log(1-null_probability)
        else:
            print("ERROR!!! Values input into logistic regressor are not 1's and 0's.")

    log_likelihood = 0
    for i in range(len(y)):
        if y[i] == 1:
            #calculate the log likelihood where likelihood = probability
            log_likelihood_i = np.log(y_pred_prob[i])
            log_likelihood += log_likelihood_i
        elif y[i] == 0:
            #calculate the log likelihood of 1-probability
            log_likelihood_i = np.log(1-y_pred_prob[i])
            log_likelihood += log_likelihood_i
        else:
            print("ERROR Values input into logistic regressor are not 1's and 0's.")

    Mcfadden_R2 = 1 - log_likelihood/null_log_likelihood
    return(Mcfadden_R2)

def q2(X,y,model=LogisticRegression()):    # This could potentially be updated to be a McFadden q2. Right not the below code below is not checked.

    loo = LeaveOneOut()
    ytests = []
    ypreds = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx] #requires arrays
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        ytests += list(y_test)
        ypreds += list(y_pred)

    rr = metrics.r2_score(ytests, ypreds)
    return(rr,ypreds)

def q2_df(X,y,model=LogisticRegression()):

    loo = LeaveOneOut()
    ytests = []
    ypreds = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] #requires arrays
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        ytests += list(y_test)
        ypreds += list(y_pred)

    rr = metrics.r2_score(ytests, ypreds)
    return(rr,ypreds)
