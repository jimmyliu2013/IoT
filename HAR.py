import sklearn
import pandas as pd
import numpy as np
import sys
import os
import argparse
import shutil
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from pandas.core.window import rolling
from scipy.ndimage.measurements import variance
from bokeh.layouts import column
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics.classification import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.sandbox.regression.kernridgeregress_class import plt_closeall
from astropy.visualization.hist import hist
from sqlalchemy.sql.expression import false

# the path of HAPT_Data_Set dir
ROOT = r"C:/Users/Administrator/Desktop/linux_879_files/879project/"

# config path and intermediate files
DATA_SET_DIR = ROOT + r"HAPT_Data_Set/RawData/"
PROCESSED_DATA_DIR = ROOT + r"Processed/"
LABLE_FILE = DATA_SET_DIR + "labels.txt"
INTERMEDIATE_DIR = PROCESSED_DATA_DIR + r"intermediate/"
FEATURE_FILE = ROOT + r"features.txt"
NORMALIZED_FEATURE_FILE = ROOT + r"normalized_features.txt"
REDUCED_FEATURE_FILE = ROOT + r"reduced_features.txt"
MY_LABELS = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND' ]

# switches
PLOT_ALL = False
DO_PCA = True
DO_LP_FILTERING = True
DO_HP_FILTERING = False  # remove gravity
DO_CROSS_VALIDATION = True

# parameters for pre-processing and features
MOVING_AVERAGE_WINDOW_SIZE = 3  # optimal
BUTTERWORTH_CUTTING_FREQUENCY = 0.2  # filter gravity
BUTTERWORTH_ORDER = 4

FEATURE_WINDOW_SIZE = 50 * 3  # 50Hz, 3seconds
OVERLAP = 0.5  # 50%

NUMBER_OF_PCA_DIMENTION = 30
TESTING_DATASET_SIZE = 0.2

# parameters for classification
NUMBER_OF_K_FOLD_CROSS_VALIDATION = 10
NUMBER_OF_CLASSIFIERS_IN_RANDOMFOREST = 120


def get_file_name_by_ids(exp_id, user_id):
    if exp_id < 10:
        exp_str = "0" + str(exp_id)
    else:
        exp_str = "" + str(exp_id)
    if user_id < 10:
        user_str = "0" + str(user_id)
    else:
        user_str = "" + str(user_id)
    acc_file = "acc_exp" + exp_str + "_user" + user_str + ".txt"
    gyro_file = "gyro_exp" + exp_str + "_user" + user_str + ".txt"
    return [acc_file, gyro_file]


def cat_acc_and_gyro(exp_id, user_id):
    acc_data = pd.read_csv(PROCESSED_DATA_DIR + get_file_name_by_ids(exp_id, user_id)[0], sep=" ", header=None)
    gyro_data = pd.read_csv(PROCESSED_DATA_DIR + get_file_name_by_ids(exp_id, user_id)[1], sep=" ", header=None)
    data = pd.concat([acc_data, gyro_data], axis=1, sort=False, ignore_index=True)
    data.to_csv(INTERMEDIATE_DIR + str(exp_id) + "_" + str(user_id) + ".txt", sep=" ", index=False, header=None)


# low-pass filter
def rolling_mean_filter(file_name):
    # print(file_name)
    name = os.path.basename(file_name)
    data = pd.read_csv(file_name, sep=" ", header=None)
    if not DO_LP_FILTERING:
        data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)
        return
    rolling_mean = data.rolling(window=MOVING_AVERAGE_WINDOW_SIZE).mean().fillna(0)
#     plt.plot(data.iloc[250:500,0], color="red", label="raw")
#     plt.plot(rolling_mean.iloc[250:500,0], color="green", label="filtered")
#     plt.title("Low-pass filter")
#     plt.xlabel("time")
#     plt.ylabel("acceleration")
#     plt.legend(['raw', 'filtered'], loc = 0, ncol = 2)
#     plt.show()
    rolling_mean.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)

    
def butterworth_filter(file_name):
    data = pd.read_csv(file_name, sep=" ", header=None)
    name = os.path.basename(file_name)
    if not DO_HP_FILTERING:
        data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)
        return
#     plt.plot(data.iloc[250:2000,0], color="red", label="raw")
    nyq = 0.5 * 50  # sampling frequency = 50Hz
    normal_cutoff = BUTTERWORTH_CUTTING_FREQUENCY / nyq
    b, a = signal.butter(BUTTERWORTH_ORDER, normal_cutoff, 'high', analog=False)
    data_0 = np.array(data.iloc[:, 0])
    data_1 = np.array(data.iloc[:, 1])
    data_2 = np.array(data.iloc[:, 2])
    out_0 = signal.filtfilt(b, a, data_0)
    out_1 = signal.filtfilt(b, a, data_1)
    out_2 = signal.filtfilt(b, a, data_2)
    data.iloc[:, 0] = out_0
    data.iloc[:, 1] = out_1
    data.iloc[:, 2] = out_2
#     plt.plot(data.iloc[250:2000,0], color="green", label="filtered")
#     plt.title("High-pass filter")
#     plt.xlabel("time")
#     plt.ylabel("acceleration")
#     plt.legend(['raw', 'filtered'], loc = 0, ncol = 2)
#     plt.show()
    data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)

    
def sort_func(file_name):
    return int(file_name.split("_")[0])


def calculate_features_for_each_column(column_data):
    mean = column_data.mean()
    max = column_data.max()
    min = column_data.min()
    med = column_data.median()
    skew = column_data.skew()
    kurt = column_data.kurt()
    std = column_data.std()
    # iqr = column_data.quantile(.75) - column_data.quantile(.25)
    # z_crossing = zero_crossing_rate(column_data)
    energy = scipy.sum(abs(column_data) ** 2) / FEATURE_WINDOW_SIZE
    f, p = scipy.signal.periodogram(column_data)
    mean_fre = scipy.sum(f * p) / scipy.sum(p)
    # max_energy_fre = np.asscalar(f[pd.DataFrame(p).idxmax()])
    # median_fre = weighted_median(f, p)
    return [mean, max, min, med, skew, kurt, std, energy, mean_fre]


def calculate_features_between_columns(column_data_1, column_data_2):
    series_1 = pd.Series(column_data_1)
    series_2 = pd.Series(column_data_2)
    corr = series_1.corr(series_2)
    return [corr]


def window_and_extract_features(data, exp_id, user_id, label, start, end):
    feature_list = []
    while True:
        if start + FEATURE_WINDOW_SIZE < end:
            row_list = [exp_id, user_id, label]
            for direction in [0, 1, 2, 3, 4, 5]:  # x,y,z axis for acc and gyro
                column_data = data.iloc[start:start + FEATURE_WINDOW_SIZE, direction]
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
                # add correlation features
                other_column = -1
                if direction == 2:
                    other_column = 0
                elif direction == 5:
                    other_column = 3
                else:
                    other_column = direction + 1
                corr = calculate_features_between_columns(column_data, data.iloc[start:start + FEATURE_WINDOW_SIZE, other_column])
                row_list.extend(corr)
            feature_list.append(row_list)
            start = (int)(start + FEATURE_WINDOW_SIZE * (1 - OVERLAP))
        else:  # if not enough data points in this window, same method to calculate features
            row_list = [exp_id, user_id, label]
            for direction in [0, 1, 2, 3, 4, 5]:  # x,y,z axis for acc and gyro
                column_data = data.iloc[start:end, direction]
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
                other_column = -1
                if direction == 2:
                    other_column = 0
                elif direction == 5:
                    other_column = 3
                else:
                    other_column = direction + 1
                corr = calculate_features_between_columns(column_data, data.iloc[start:end, other_column])
                row_list.extend(corr)
            feature_list.append(row_list)
            break
    result = pd.DataFrame(feature_list)
    # print(feature_list)
    return result


#*****************************************#
#      1.filter all the raw data file     #
#*****************************************#
def filter_data():
    files = os.listdir(DATA_SET_DIR)
    for file in files:
        if not file.startswith("labels"):
            file_name = DATA_SET_DIR + file
            # print(file_name)
            rolling_mean_filter(file_name)
            if file.startswith("acc"):  # gravity only exists in acc data
                # use processed data
                butterworth_filter(PROCESSED_DATA_DIR + file)


#*****************************************#
#        2.cat acc and gyro data          #
#*****************************************#
def catenate_data():
    data = pd.read_csv(LABLE_FILE, sep=" ", header=None)
    for (idx, row) in data.iterrows():
        if not os.path.exists(INTERMEDIATE_DIR + str(row[0]) + "_" + str(row[1]) + ".txt"):
            cat_acc_and_gyro(row[0], row[1])


#*****************************************#
#      3.feature extraction and label     #
#*****************************************#
def extract_features():
    label_data = pd.read_csv(LABLE_FILE, sep=" ", header=None)
    new_data = pd.DataFrame()
    for (idx, row) in label_data.iterrows():
        data = pd.read_csv(INTERMEDIATE_DIR + str(row[0]) + "_" + str(row[1]) + ".txt", sep=" ", header=None)
        exp_id = row[0]
        user_id = row[1]
        start = row[3]
        end = row[4]
        label = row[2]
        sub_dataframe = window_and_extract_features(data, exp_id, user_id, label, start, end)
        new_data = new_data.append(sub_dataframe)
    print("feature matrix shape before PCA: " + str(new_data.shape))  # shape of raw features
    new_data.to_csv(FEATURE_FILE, sep=" ", index=False, header=None)


#*****************************************#
#         4.feature normalization         #
#*****************************************#
def normalize_data():
    features = pd.read_csv(FEATURE_FILE, sep=" ", header=None)
    # print(features.head(5))
    for column in features.columns[3:]:
        col = features[[column]].values.astype(float)
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        normalized_col = min_max_scaler.fit_transform(col)
        features.iloc[:, column] = normalized_col
    features.to_csv(NORMALIZED_FEATURE_FILE, sep=" ", index=False, header=None)


#*****************************************#
#           5.feature reduction           #
#*****************************************#
def pca():
    features = pd.read_csv(NORMALIZED_FEATURE_FILE, sep=" ", header=None)
    if not DO_PCA:
        features.to_csv(REDUCED_FEATURE_FILE, sep=" ", index=False, header=None)
        return
    without_label = features.iloc[:, 3:]
    # from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    pca = PCA().fit(without_label)
    if PLOT_ALL:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.show()
    pca = PCA(n_components=NUMBER_OF_PCA_DIMENTION)
    pca.fit(without_label)
    X_pca = pca.transform(without_label)
    df = pd.DataFrame(X_pca)
    new_data = pd.concat([features.iloc[:, :3], df], axis=1, sort=False, ignore_index=True)  # add labels
    new_data.to_csv(REDUCED_FEATURE_FILE, sep=" ", index=False, header=None)


def plot_report(y_test, test_predict, title):
    precision = precision_score(y_test, test_predict, average=None)
    recall = recall_score(y_test, test_predict, average=None)
    f1 = f1_score(y_test, test_predict, average=None)
    plt.tight_layout(pad=0) 
    plt.plot(precision, color="red")
    plt.plot(recall, color="green")
    plt.plot(f1, color="blue")
    plt.margins(x=0)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.title(title)
    plt.legend(["precision", "recall", "f1-score"])
    plt.xticks(np.arange(0, 12, step=1), MY_LABELS, rotation=60, fontsize=6)
    plt.show()

    
def plot_label_distribution(y_train):
        # print(y_train.value_counts())
        label_distribution = y_train.value_counts().reset_index()
        sorted = label_distribution.sort_values(['index'])
        sorted.set_index('index').plot(kind='bar')
        plt.title("Distribution of labels in training data")
        plt.xlabel("")
        plt.ylabel("number of samples")
        plt.gcf().subplots_adjust(bottom=0.5)
        plt.gca().get_legend().remove()
        plt.xticks(np.arange(0, 12, step=1), MY_LABELS, rotation=60, fontsize=6)
        plt.show()


#*****************************************#
#          5.classification: KNN          #
#*****************************************#
def KNN(X_train, X_test, y_train, y_test):
    print(X_train.shape)
    """ # use this to choose K
    k = []
    score = []
    for i in range(0, 50):
        if i%2 != 0:
            print(i)
            model = KNeighborsClassifier(n_neighbors=i)
            scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True, random_state=0), scoring='accuracy')
            k.append(i)
            score.append(scores.mean())
     
    plt.plot(k, score)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')
    plt.show()
    return
    """
    
    print("################# KNN #################")
    model = KNeighborsClassifier(n_neighbors=9)
    if DO_CROSS_VALIDATION:
        scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("KNN cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "KNN")
    print("report for KNN: ")
    report = sklearn.metrics.classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS)
    print(report)
    print("KNN overall accuracy: " + str(sklearn.metrics.accuracy_score(y_test, test_predict)))
    # print(confusion_matrix(y_test, test_predict))


#*****************************************#
#          6.classification: SVM          #
#*****************************************#
def NaiveBayes(X_train, X_test, y_train, y_test):
    print(X_train.shape)
    model = GaussianNB()
    print("################# NaiveBayes #################")
    if DO_CROSS_VALIDATION:
        scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("NaiveBayes cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "Naive Bayes")
    print("report for NaiveBayes: ")
    print(sklearn.metrics.classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    print("NaiveBayes overall accuracy: " + str(sklearn.metrics.accuracy_score(y_test, test_predict)))


#*****************************************#
#          7.classification: SVM          #
#*****************************************#
def SVM(X_train, X_test, y_train, y_test):
    print(X_train.shape)
    model = sklearn.svm.SVC(kernel='linear', C=1000, decision_function_shape='ovo')
    
    """ # gridsearch to find hyperparameters
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]}
                   ]
    clf = sklearn.model_selection.GridSearchCV(model1, tuned_parameters, cv=KFold(n_splits=10, shuffle=True, random_state=0))
    clf.fit(without_labels, labels)
    print(clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return
    """
    print("################# SVM #################")
    if DO_CROSS_VALIDATION:
        scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("SVM cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "SVM")
    print("report for SVM: ")
    print(sklearn.metrics.classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    print("SVM overall accuracy: " + str(sklearn.metrics.accuracy_score(y_test, test_predict)))
    # print(sklearn.metrics.confusion_matrix(y_test, test_predict))


#*****************************************#
#       8.classification: RandomForest    #
#*****************************************#
def RandomForest(X_train, X_test, y_train, y_test):
    print(X_train.shape)
    model = RandomForestClassifier(n_estimators=NUMBER_OF_CLASSIFIERS_IN_RANDOMFOREST, criterion='entropy', max_features='log2')
    """ # gridsearch to find hyperparameters
    tuned_parameters = {'n_estimators': [50, 80, 100, 120, 150, 200],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'criterion' :['gini', 'entropy']
                  }
    clf = sklearn.model_selection.GridSearchCV(model, tuned_parameters, cv=KFold(n_splits=10, shuffle=True, random_state=0), scoring='accuracy')
    clf.fit(X_train, y_train)
    print("-------------------------------------")
    print(clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return
    """
    print("################# RandomForest #################")
    if DO_CROSS_VALIDATION:
        scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("RandomForest cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "Random Forest")
    print("report for RandomForest: ")
    print(sklearn.metrics.classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    print("RandomForest overall accuracy: " + str(sklearn.metrics.accuracy_score(y_test, test_predict)))
    
    """ # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    """


def init():
    clean()
    os.mkdir(PROCESSED_DATA_DIR)
    os.mkdir(INTERMEDIATE_DIR)


def clean():
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    if os.path.exists(FEATURE_FILE):
        os.remove(FEATURE_FILE)
    if os.path.exists(NORMALIZED_FEATURE_FILE):
        os.remove(NORMALIZED_FEATURE_FILE)
    if os.path.exists(REDUCED_FEATURE_FILE):
        os.remove(REDUCED_FEATURE_FILE)


def main():
    print("start...............................")
    init()
    print("filter data")
    filter_data()
    catenate_data()
    print("extract features")
    extract_features()
    print("normalize features")
    normalize_data()
    pca()

    labels = None
    without_labels = None
#   without_labels = pd.read_csv(r"C:/Users/Administrator/Desktop/linux_879_files/879project/HAPT_Data_Set/Train/X_train.txt", sep=" ", header=None)
#   labels = pd.read_csv(r"C:/Users/Administrator/Desktop/linux_879_files/879project/HAPT_Data_Set/Train/y_train.txt", sep=" ", header=None)
        
    data = pd.read_csv(REDUCED_FEATURE_FILE, sep=" ", header=None)
    data = data.iloc[:, 2:]  # remove user id and experiment id
    labels = data.iloc[:, 0]
    without_labels = data.iloc[:, 1:]
    # split data
    X_train, X_test, y_train, y_test = train_test_split(without_labels, labels, test_size=TESTING_DATASET_SIZE)
    # plot_label_distribution(y_train)
        
    KNN(X_train, X_test, y_train, y_test)
    NaiveBayes(X_train, X_test, y_train, y_test)
    SVM(X_train, X_test, y_train, y_test)
    RandomForest(X_train, X_test, y_train, y_test)

    print("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    main()

# 1 WALKING           
# 2 WALKING_UPSTAIRS  
# 3 WALKING_DOWNSTAIRS
# 4 SITTING           
# 5 STANDING          
# 6 LAYING            
# 7 STAND_TO_SIT      
# 8 SIT_TO_STAND      
# 9 SIT_TO_LIE        
# 10 LIE_TO_SIT        
# 11 STAND_TO_LIE      
# 12 LIE_TO_STAND
