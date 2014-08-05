# creates dummy variables

import pandas as pd
import numpy as np

df = pd.read_csv('../data/titanic_train.csv')

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)



df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# create dummy variables to eliminate problem of ordering for categorical values
pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = cols[1:2] + cols[0:1] + cols[2:]

df = df[cols]




train_data = df.values




from sklearn.svm import SVC

model = SVC(kernel='linear')
model = model.fit(train_data[0::,1::], train_data[0::,0])



df_test = pd.read_csv('../data/titanic_test.csv')

df_test.info()

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)



fare_means = df.pivot_table('Fare', rows='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)



df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)

# create dummy variables to eliminate problem of ordering for categorical values
pd.get_dummies(df_test['Embarked'], prefix='Embarked').head(10)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],
                axis=1)

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values


output = model.predict(test_data)


result = np.c_[test_data[:,0].astype(int), output.astype(int)]


df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('../results/titanic_two.csv', index=False)




#1027 	↓47 	savarin 	0.77033 	10 	Mon, 04 Aug 2014 14:37:15 (-2.3h)
#Your Best Entry
#Your submission scored 0.77033, which is not an improvement of your best score. Keep trying!
