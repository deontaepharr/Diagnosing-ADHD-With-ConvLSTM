import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

data_filepath = "/home/deopha32/ADHD-FMRI/References/model_data.csv"
training_filepath = "/home/deopha32/ADHD-FMRI/Data/training_data_{}"
validation_filepath = "/home/deopha32/ADHD-FMRI/Data/validatation_data_{}"
testing_filepath = "/home/deopha32/ADHD-FMRI/Data/testing_data"

model_csv = data_filepath
model_data = pd.read_csv(model_csv)

X = model_data['Image']
y = model_data['DX']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

count = 1
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    model_train_data = pd.DataFrame(pd.concat([X_train_fold, y_train_fold], axis=1))
    model_val_data = pd.DataFrame(pd.concat([X_val_fold, y_val_fold], axis=1))
    
    model_train_data.to_csv(training_filepath.format(count), index=False)
    model_val_data.to_csv(validation_filepath.format(count), index=False)
    
    count += 1

model_test_data = pd.DataFrame(pd.concat([X_test, y_test], axis=1))
model_test_data.to_csv(testing_filepath, index=False)