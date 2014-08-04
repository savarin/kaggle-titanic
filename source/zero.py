
import pandas as pd
import numpy as np

df = pd.read_csv('../datasets/titanic_train.csv')


# review first ten rows
df.head(10)

# identify columns to process
df.info()


# drop columns Ticket and Cabin
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# drop NA values
df = df.dropna()


# review unique values, create numerical column for Sex
df['Sex'].unique()
df['Gender'] = df['Sex'].map({'female': 0, 'male':1})

# review unique values, create numerical column for Embarked
df['Embarked'].unique()
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3})

# drop Name, Sex, Embarked
df = df.drop(['Sex', 'Embarked'], axis=1)

# relabel columns such that Survived at start

cols = df.columns.tolist()
cols = cols[1:2] + cols[0:1] + cols[2:]

df = df[cols]


# final review of df, ensure (1) Survived first column (2) no NA values (3) numerical
df.head(10)
df.info()


# create array of values

train_data = df.values


# apply SVM model

from sklearn import svm

model = svm.SVC(kernel='linear')
model = model.fit(train_data[0::,1::], train_data[0::,0])


# apply to test data, similar cleaning method

df_test = pd.read_csv('../datasets/titanic_test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test = df_test.dropna()

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values


# apply model on test data
output = model.predict(test_data)

# create array from results
result = np.c_[test_data[:,0].astype(int), output.astype(int)]

# print to output file
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('../results/titanic_one.csv', index=False)

# problem - output smaller than required for submission
df_result.shape
