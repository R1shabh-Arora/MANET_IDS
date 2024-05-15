import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import warnings
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split
import os

#  get all the command line args
import sys

things_to_remove = sys.argv[1:]


# A function which loads the dataset and preprocesses it
def load_dataset(datasetPath, things_to_remove=[]):
    # Load the dataset
    megaDataset = pd.read_csv(datasetPath)
    # Drop rows with missing values
    megaDataset = megaDataset.dropna()
    # Drop columns which are not needed
    if len(things_to_remove) > 0:
        megaDataset = megaDataset.drop(columns=things_to_remove)
    # Force all non numerical columns to be strings
    nonNumericalColumn = megaDataset.columns.difference(megaDataset.select_dtypes(include='number').columns)
    for col in nonNumericalColumn:
        megaDataset[col] = megaDataset[col].astype(str)
    # Drop label col
    try:
        megaDataset.drop(columns=["label"], inplace=True)
    except: 
        pass
    return megaDataset


# For every numerical column, create a new scaler. Do not apply the scaler yet.
def get_scalers(df, numericalCols):
    scalers = {}
    for col in numericalCols:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array(df[col].values).reshape(-1, 1))
        scalers[col] = scaler
    return scalers

def apply_scalers(df, scalers):
    for col in scalers:
        df[col] = scalers[col].transform(np.array(df[col].values).reshape(-1, 1))
    return df

# Create the label encoders
def get_onehot_encoder(df, nonNumericalColumns):
    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
    ohe.fit(df[nonNumericalColumns])
    return ohe

# Apply the label encoders
def apply_onehot_encoder(df, onehot_encoder: preprocessing.OneHotEncoder, nonNumericalColumns):
    onehotEncoded = onehot_encoder.transform(df[nonNumericalColumns]).toarray()
    onehotEncoded = pd.DataFrame(onehotEncoded, columns=onehot_encoder.get_feature_names_out(onehot_encoder.feature_names_in_))
    df = df.drop(columns=nonNumericalColumns)
    df = pd.concat([df, onehotEncoded], axis=1)
    return df

def get_X_Y(df):
    # Y is all columns starting with attack_cat
    Y = df.filter(regex='attack_cat.*')
    # X is all columns except Y
    X = df.drop(columns=Y.columns)
    # Sort cols alphabetically
    X = X.reindex(sorted(X.columns), axis=1)
    Y = Y.reindex(sorted(Y.columns), axis=1)
    # Conver Y to a 1d numpy array of indexes
    Y = np.argmax(Y.values, axis=1)
    return X, Y

def get_categorical_columns(df):
    return df.columns.difference(df.select_dtypes(include='number').columns)

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


trainset = load_dataset("UNSW_NB15_training-set.csv", things_to_remove=things_to_remove)
testset = load_dataset("cleaned_UNSW_NB15_testing-set.csv", things_to_remove=things_to_remove)

categorical_columns = ['attack_cat', 'proto', 'service', 'state']
categorical_columns = trainset.columns.intersection(categorical_columns)
numerical_columns = trainset.columns.difference(categorical_columns)

scalers = get_scalers(trainset, numerical_columns)

trainset = apply_scalers(trainset, scalers)
testset = apply_scalers(testset, scalers)

label_encoders = get_onehot_encoder(trainset, categorical_columns)

trainset = apply_onehot_encoder(trainset, label_encoders, categorical_columns)
testset = apply_onehot_encoder(testset, label_encoders, categorical_columns)

# Shuffle the dataset
trainset = trainset.sample(frac=1).reset_index(drop=True)
testset = testset.sample(frac=1).reset_index(drop=True)

print(f"Trainset has shape {trainset.shape}")
print(f"Testset has shape {testset.shape}")

X_train, Y_train = get_X_Y(trainset)
X_test, Y_test = get_X_Y(testset)

print("Training the model now")

# SVM
model = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, Y_train)
 
predicted = model.predict(X_test)

def custom_confusion_matrix(Y_predicted, Y_true, num_classes):
    mat = np.zeros((num_classes, num_classes))
    for i in range(len(Y_predicted)):
        mat[Y_true[i]][Y_predicted[i]] += 1
    return mat

def get_f1s(cm):
    f1 = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        f1[i] = 2 * tp / (2 * tp + fp + fn)
    return f1

def get_macro_f1(f1s):
    return np.mean(f1s)

def get_accuracies(cm):
    return np.diagonal(cm) / np.sum(cm, axis=1)

def get_accuracy(accs):
    return np.mean(accs)

label_names = [l for l in label_encoders.get_feature_names_out() if "attack" in l]

label_names = [l.split("_")[2] for l in label_names]

cm = custom_confusion_matrix(predicted, Y_test, len(label_names))

# Seaborn heatmap
plt.figure(figsize=(10, 7))
sb.heatmap(cm, annot=True, xticklabels=label_names, yticklabels=label_names, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("reports/"+"_".join(things_to_remove) + ".png")

f1s = get_f1s(cm)
f1 = get_macro_f1(f1s)

accuaccies = get_accuracies(cm)
accuaccy = get_accuracy(accuaccies)

print(f"Macro F1: {f1}")
print(f"Accuracy: {accuaccy}")
print(f"F1s: {f1s}")
print(f"Accuracies: {accuaccies}")

total_df = pd.DataFrame(columns=['F1', 'Accuracy'])

# Create a row for every class
for i in range(len(label_names)):
    total_df.loc[label_names[i]] = [f1s[i], accuaccies[i]]
total_df.loc['Macro_Avg'] = [f1, accuaccy]

total_df.to_csv("reports/"+"_".join(things_to_remove) + ".csv")

#report  = metrics.classification_report(Y_test, predicted, target_names=label_encoders["attack_cat "], output_dict=True)
#df_report = pd.DataFrame(report).transpose()
#df_report.to_csv("_".join(things_to_remove) + ".csv")


# Print that the results are saved in this file
#print("_".join(things_to_remove) + ".csv")