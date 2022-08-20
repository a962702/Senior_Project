############################################################
# Download dataset and process PCA if needed,              #
# then save to 'tmp_data.npz'                              #
#                                                          #
# Usage: process_dataset.py <dataset name> <PCA percent>   #
############################################################

import sys
import pandas as pd
from sklearn import preprocessing
import numpy as np
from pcaa import pcaa
from sklearn.model_selection import train_test_split

if len(sys.argv) < 3:
    print('-> Usage: process_dataset.py <dataset name> <PCA percent>')
    sys.exit()

name = str(sys.argv[1])
pca_percent = str(sys.argv[2])

print('-> Choose Dataset: ' + name)
# Step 1. Download dataset from GitHub
print('--> Step 1. Download dataset from GitHub')
df1 = pd.read_csv('https://github.com/a962702/Senior_Project/raw/main/csv/' + name + '/ransomware/labeled_name/merged.csv')
df2 = pd.read_csv('https://github.com/a962702/Senior_Project/raw/main/csv/' + name + '/normal/labeled_normal/merged.csv')
dataset = pd.concat([df1, df2])
print('---> Shape of dataset',dataset.shape)

# Step 2. Label Encoding
print('--> Step 2. Label Encoding')
#### label coding for nominal values
def label_coding(label):
    dataset[label]= label_encoder.fit_transform(dataset[label]) 
    dataset[label].unique()
### label encoding
label_encoder = preprocessing.LabelEncoder() 
label_coding('Flow ID')
label_coding('Src IP')
label_coding('Dst IP')
label_coding('Timestamp')
label_coding('Label')

# Step 3. Extract features
print('--> Step 3. Extract features')
X=dataset.iloc[:,:-1]
X=X.values
print("+-inf",sum(np.isinf(X)))
print("inf",sum(np.isposinf(X)))
print("-inf",sum(np.isneginf(X)))
print("nan",sum(np.isnan(X)))
print("fin",sum(np.isfinite(X)))
X = np.where(np.isnan(X), 0, X)
X = np.where(np.isposinf(X), 0, X)
X = np.where(np.isneginf(X), 0, X)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Step 4. PCA processing (Optional)
print('--> Step 4. PCA processing')
if pca_percent != 'origin':
    pca_percent = float(pca_percent)
    print('---> Processing PCA with ' + str(pca_percent * 100) + '%')
    fin = pcaa(X, pca_percent)
    X = fin[0]
else:
    print('---> Skip PCA Processing')

# Step 5. Split train and test data
print('--> Step 5. Split train and test data')
Y = dataset['Label'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

# Step 6. Saving data to file
print('--> Step 6. Saving data to file')
np.savez('tmp_data.npz', X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test)

print('-> Done')
