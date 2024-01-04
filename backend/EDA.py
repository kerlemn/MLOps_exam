import pandas as pd
import numpy as np

from helper import get_training_data, load_user_feedback
# from knn import knn

from sklearn.linear_model import LogisticRegression
from sklearn.tree         import DecisionTreeClassifier
from sklearn.svm          import LinearSVC
from sklearn.naive_bayes  import GaussianNB
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import StandardScaler

import graphviz
from graphviz import Source
from sklearn import tree

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from time import time


import warnings
warnings.filterwarnings("ignore")




def get_data():
    df = pd.read_csv('backend/user_data_example.csv', sep='\t')
    cols = df.columns.tolist()
    cols.remove('title')
    cols.remove('score')

    titles = df['title']
    x = df[cols].values
    y = df['score'].values

    return cols, x, y

def train_models(to_train, cv, X, y):
    # Split the data in train and test set

    models        = []
    accuracies    = []
    f1_scores     = []
    train_times   = []
    predict_times = []

    for model in to_train:
        k_fold_accuracy_mean = []
        k_fold_f1_scores     = []
        
        for _ in range(cv):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipe = model
            # Train the model and keep the times
            start_train = time()
            clf = pipe.fit(X_train, y_train)
            end_train = time()

            # Make the prediction and keep the times
            start_pred = time()
            pred = pipe.predict(X_test)
            end_pred = time()

            # Calculate the mean of the accuracies for the algorithm
            k_fold_accuracy_mean.append(accuracy_score(y_test, pred))
            # Calculate the mean of the f1-scores for the algorithm
            k_fold_f1_scores    .append(f1_score(y_test, pred))

            # datasets = [train_test_split(X, y, test_size=0.2) for _ in range(cv)]

            # k_fold_f1_scores     = np.mean([f1_score(y_test, DecisionTreeClassifier(max_depth=3).fit(X_train, y_train).predict(X_test)) for X_train, X_test, y_train, y_test in datasets])

            # Append the informations
        models       .append(clf)
        accuracies   .append(round( np.mean(k_fold_accuracy_mean), 3 ))
        f1_scores    .append(round( np.mean(k_fold_f1_scores), 3 ))
        train_times  .append(round(end_train - start_train, 5))
        predict_times.append(round(end_pred - start_pred, 5))

    return models, accuracies, f1_scores, train_times, predict_times

def decision_tree_changed(X, y, titles):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model and keep the times
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    # Make the prediction and keep the times
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    prob = [round(p, 3) for p in prob]
    
    user_info  = np.mean(X_train[y_train], axis=0) 
    
    true_pages = np.array([page if val 
                           else "None" 
                           for page, val in zip(X_test, pred)])

    cosines  = np.array([round(cosine(user_info, page), 3) if page != "None"
                         else "None" 
                         for page in true_pages])
    MSE      = np.array([round(np.mean((user_info-page)**2), 3) if page != "None" 
                         else "None" 
                         for page in true_pages])

    df = pd.DataFrame(np.array([cosines, y_test, pred, prob]).T, columns=["cosine", "y_test", "pred", "prob"])
    df = df[df['cosine'] != "None"]
    df = df.sort_values(by=['cosine'])
    print(df)
    
    df = pd.DataFrame(np.array([MSE, y_test, pred, prob]).T, columns=["MSE", "y_test", "pred", "prob"])
    df = df[df['MSE'] != "None"]
    df = df.sort_values(by=['MSE'])
    print()
    print(df)

    mix = np.array([round(float(cos) + float(mse), 3) if cos != "None" and mse != "None"
                    else "None"
                    for cos, mse in zip(cosines, MSE)])
    df = pd.DataFrame(np.array([mix, y_test, pred, prob]).T, columns=["mix", "y_test", "pred", "prob"])
    df = df[df['mix'] != "None"]
    df = df.sort_values(by=['mix'])
    print()
    print(df)
    

    
    






def plot_times(df, model):
    df[model].plot(kind='line', figsize=(8, 4), title="Algorithms times")
    plt.gca().spines[['top', 'right']].set_visible(False)

def enlarge_dataset(X, y):
    enlarged_data          = [(X, y)]
    X_enlarged, y_enlarged = X, y

    for n in range(1000, 11000, 1000):
        idxs = np.random.choice(range(len(X)), 1000)
        X_enlarged = np.array(X_enlarged.tolist() + X[idxs].tolist())
        y_enlarged = np.array(y_enlarged.tolist() + y[idxs].tolist())

        enlarged_data.append((X_enlarged, y_enlarged))

    return enlarged_data

def get_times(model, enlarged_data):
    train = []
    pred  = []

    for X_enlarged, y_enlarged in enlarged_data:
        train_times = []
        if model == "KNN":
            train_times   += [0]
        else:
            for _ in range(10):
                start_train   = time()
                clf           = model.fit(X_enlarged, y_enlarged)
                end_train     = time()

                train_times   += [end_train - start_train]

        train.append(np.mean(train_times))

        pred_times  = []
        if model == "KNN":
            for _ in range(10):
                start_predict = time()
                knn(X_enlarged, y_enlarged, X_enlarged)
                end_predict   = time()

                pred_times    += [end_predict - start_predict]
        else:
            for _ in range(10):
                start_predict = time()
                clf.predict(X_enlarged)
                end_predict   = time()

                pred_times    += [end_predict - start_predict]
        
        pred.append(np.mean(pred_times))

    return train, pred

#################
### Load data ###
#################
user = "Stefano"
X, y, columns = get_training_data(user)
y = y == "True"
titles, _, _ = get_data()


clf = LogisticRegression(max_iter=3000).fit(X, y)
print(clf.coef_[0])
clf.coef_[0] = 0.9 * np.array(clf.coef_[0]) + np.array([0.5] * len(clf.coef_[0])) * 0.1
clf.coef_[0] = clf.coef_[0].tolist()
print(clf.coef_[0])

decision_tree_changed(X, y, titles)

#######################
### Train the model ###
#######################
to_train = [LogisticRegression(), DecisionTreeClassifier(), LinearSVC(), GaussianNB()]
models, accuracies, f1_scores, train_times, predict_times = train_models(to_train, 10, X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# time1 = time()
# results = knn(X_train, y_train, X_test)
# time2 = time()

# y_hat = np.zeros(len(y_test))
# y_hat[[results]] = 1

# train_times  .append(0)
# predict_times.append(round(time2 - time1, 3))
# accuracies   .append(round(accuracy_score(y_test, y_hat), 5))
# f1_scores    .append(round(f1_score(y_test, y_hat), 5))

#########################
### Print the results ###
#########################
algorithm_names=["Logistic Regression", "Decision Tree", "Support Vector Classifier", "Gaussian Naive Bayes", "KNN"]
data = np.array([accuracies, f1_scores, train_times, predict_times]).T

results_df = pd.DataFrame(data,
                          columns=["Accuracy", "f1-score", "train-time", "predict-time"],
                          index=algorithm_names)
print(results_df)

plt.figure(figsize=(30, 30))
tree.plot_tree(models[1],filled=True, feature_names=titles, class_names=["False", "True"])  
plt.savefig('tree.png',format='png',bbox_inches = "tight")

plt.cla()
plt.figure()
plt.bar(range(1, len(titles) + 1), models[1].feature_importances_)
plt.xticks(range(1, len(titles) + 1), titles, rotation=90)
plt.savefig('barplot.png',format='png',bbox_inches = "tight")

#########################
### Check scalability ###
#########################
algorithm_names=["Logistic Regression", "Decision Tree", "Support Vector Classifier", "Gaussian Naive Bayes"]
enlarged_data = enlarge_dataset(X, y)
LR_times, LR_pred_times     = get_times(LogisticRegression(), enlarged_data)
DT_times, DT_pred_times     = get_times(DecisionTreeClassifier(max_depth=3), enlarged_data)
SVC_times, SVC_pred_times   = get_times(LinearSVC(), enlarged_data)
GNB_times, GNB_pred_times   = get_times(GaussianNB(), enlarged_data)
# KNN_times, KNN_pred_times   = get_times("KNN", enlarged_data)

train_data = np.array([LR_times, DT_times, SVC_times, GNB_times]).T
pred_data  = np.array([LR_pred_times, DT_pred_times, SVC_pred_times, GNB_pred_times]).T

train_times_df = pd.DataFrame(train_data,
                              columns=algorithm_names,
                              index=range(0 + len(X), 11000 + len(X), 1000))
pred_times_df = pd.DataFrame(pred_data,
                             columns=algorithm_names,
                             index=range(0 + len(X), 11000 + len(X), 1000))
plt.cla()
for model in algorithm_names:
   plot_times(train_times_df, model)

plt.grid()
plt.title("Train")
plt.legend()
plt.savefig("train.jpg")

plt.cla()
for model in algorithm_names:
   plot_times(pred_times_df, model)

plt.grid()
plt.title("Predict")
plt.legend()
plt.savefig("pred.jpg")