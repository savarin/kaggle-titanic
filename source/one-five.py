# introduces Pipeline + Imputation, may break up so
# first only introduces Imputation, second choice mean/median to impute

import pandas as pd
import numpy as np

df = pd.read_csv('../data/titanic_train.csv')

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#age_mean = df['Age'].mean()
#df['Age'] = df['Age'].fillna(age_mean)

from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]

df = df[cols]

# Because of the following bug we cannot use NaN as the missing
# value marker, use a negative value as marker instead:
# https://github.com/scikit-learn/scikit-learn/issues/3044

# fill NA values in Age with -1
df = df.fillna(-1)


train_data = df.values



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

imputer = Imputer(strategy='mean', missing_values=-1)

classifier = RandomForestClassifier(n_estimators=100)

pipeline = Pipeline([
    ('imp', imputer),
    ('clf', classifier),
])

parameter_grid = {
    'imp__strategy': ['mean', 'median'],
    'clf__max_features': [0.5, 1],
    'clf__max_depth': [5, None],
}



sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)

grid_search.best_score_

grid_search.best_params_



# choose parameter with best score

model = RandomForestClassifier(n_estimators = 100, max_features=0.5, max_depth=None)
model = model.fit(train_data[0::,1::],train_data[0::,0])



df_test = pd.read_csv('../data/titanic_test.csv')

df_test.info()

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)


fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)



df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)

pd.get_dummies(df_test['Embarked'], prefix='Embarked').head(10)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],
                axis=1)

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values


output = model.predict(test_data)

result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('../results/titanic_1-4.csv', index=False)



"""

 mean: 0.74747, std: 0.03122, params: {'C': 0.01, 'gamma': 0.01},
 mean: 0.74747, std: 0.03122, params: {'C': 0.01, 'gamma': 0.1},
 mean: 0.74747, std: 0.03122, params: {'C': 0.01, 'gamma': 1.0},
 mean: 0.74747, std: 0.03122, params: {'C': 0.01, 'gamma': 10.0},
 mean: 0.74747, std: 0.03122, params: {'C': 0.01, 'gamma': 100.0},
 mean: 0.78563, std: 0.00884, params: {'C': 0.1, 'gamma': 0.01},
 mean: 0.78563, std: 0.00884, params: {'C': 0.1, 'gamma': 0.1},
 mean: 0.78563, std: 0.00884, params: {'C': 0.1, 'gamma': 1.0},
 mean: 0.78563, std: 0.00884, params: {'C': 0.1, 'gamma': 10.0},
 mean: 0.78563, std: 0.00884, params: {'C': 0.1, 'gamma': 100.0},
 mean: 0.79012, std: 0.00840, params: {'C': 1.0, 'gamma': 0.01},
 mean: 0.79012, std: 0.00840, params: {'C': 1.0, 'gamma': 0.1},
 mean: 0.79012, std: 0.00840, params: {'C': 1.0, 'gamma': 1.0},
 mean: 0.79012, std: 0.00840, params: {'C': 1.0, 'gamma': 10.0},
 mean: 0.79012, std: 0.00840, params: {'C': 1.0, 'gamma': 100.0},
 mean: 0.79349, std: 0.01789, params: {'C': 10.0, 'gamma': 0.01},
 mean: 0.79349, std: 0.01789, params: {'C': 10.0, 'gamma': 0.1},
 mean: 0.79349, std: 0.01789, params: {'C': 10.0, 'gamma': 1.0},
 mean: 0.79349, std: 0.01789, params: {'C': 10.0, 'gamma': 10.0},
 mean: 0.79349, std: 0.01789, params: {'C': 10.0, 'gamma': 100.0},
 mean: 0.79461, std: 0.01672, params: {'C': 100.0, 'gamma': 0.01},
 mean: 0.79461, std: 0.01672, params: {'C': 100.0, 'gamma': 0.1},
 mean: 0.79461, std: 0.01672, params: {'C': 100.0, 'gamma': 1.0},
 mean: 0.79461, std: 0.01672, params: {'C': 100.0, 'gamma': 10.0},
 mean: 0.79461, std: 0.01672, params: {'C': 100.0, 'gamma': 100.0}



 grid_search.fit(train_data[0::,1::], train_data[0::,0])
Fitting 3 folds for each of 25 candidates, totalling 75 fits
[GridSearchCV] C=0.01, gamma=0.01 ..............................................
[GridSearchCV] ..................... C=0.01, gamma=0.01, score=0.764310 -   0.6s
[GridSearchCV] C=0.01, gamma=0.01 ..............................................
[GridSearchCV] ..................... C=0.01, gamma=0.01, score=0.703704 -   0.4s
[GridSearchCV] C=0.01, gamma=0.01 ..............................................
[GridSearchCV] ..................... C=0.01, gamma=0.01, score=0.774411 -   0.4s
[GridSearchCV] C=0.01, gamma=0.1 ...............................................
[GridSearchCV] ...................... C=0.01, gamma=0.1, score=0.764310 -   0.6s
[GridSearchCV] C=0.01, gamma=0.1 ...............................................
[GridSearchCV] ...................... C=0.01, gamma=0.1, score=0.703704 -   0.4s
[GridSearchCV] C=0.01, gamma=0.1 ...............................................
[GridSearchCV] ...................... C=0.01, gamma=0.1, score=0.774411 -   0.5s
[GridSearchCV] C=0.01, gamma=1.0 ...............................................
[GridSearchCV] ...................... C=0.01, gamma=1.0, score=0.764310 -   0.6s
[GridSearchCV] C=0.01, gamma=1.0 ...............................................
[GridSearchCV] ...................... C=0.01, gamma=1.0, score=0.703704 -   0.4s
[GridSearchCV] C=0.01, gamma=1.0 ...............................................
[GridSearchCV] ...................... C=0.01, gamma=1.0, score=0.774411 -   0.4s
[GridSearchCV] C=0.01, gamma=10.0 ..............................................
[GridSearchCV] ..................... C=0.01, gamma=10.0, score=0.764310 -   0.6s
[GridSearchCV] C=0.01, gamma=10.0 ..............................................
[GridSearchCV] ..................... C=0.01, gamma=10.0, score=0.703704 -   0.4s
[GridSearchCV] C=0.01, gamma=10.0 ..............................................
[GridSearchCV] ..................... C=0.01, gamma=10.0, score=0.774411 -   0.4s
[GridSearchCV] C=0.01, gamma=100.0 .............................................
[GridSearchCV] .................... C=0.01, gamma=100.0, score=0.764310 -   0.6s
[GridSearchCV] C=0.01, gamma=100.0 .............................................
[GridSearchCV] .................... C=0.01, gamma=100.0, score=0.703704 -   0.4s
[GridSearchCV] C=0.01, gamma=100.0 .............................................
[GridSearchCV] .................... C=0.01, gamma=100.0, score=0.774411 -   0.4s
[GridSearchCV] C=0.1, gamma=0.01 ...............................................
[GridSearchCV] ...................... C=0.1, gamma=0.01, score=0.777778 -  26.7s
[GridSearchCV] C=0.1, gamma=0.01 ...............................................
[GridSearchCV] ...................... C=0.1, gamma=0.01, score=0.797980 -   5.5s
[GridSearchCV] C=0.1, gamma=0.01 ...............................................
[GridSearchCV] ...................... C=0.1, gamma=0.01, score=0.781145 -  25.3s
[GridSearchCV] C=0.1, gamma=0.1 ................................................
[GridSearchCV] ....................... C=0.1, gamma=0.1, score=0.777778 -  26.3s
[GridSearchCV] C=0.1, gamma=0.1 ................................................
[GridSearchCV] ....................... C=0.1, gamma=0.1, score=0.797980 -   5.5s
[GridSearchCV] C=0.1, gamma=0.1 ................................................
[GridSearchCV] ....................... C=0.1, gamma=0.1, score=0.781145 -  25.7s
[GridSearchCV] C=0.1, gamma=1.0 ................................................
[GridSearchCV] ....................... C=0.1, gamma=1.0, score=0.777778 -  26.8s
[GridSearchCV] C=0.1, gamma=1.0 ................................................
[GridSearchCV] ....................... C=0.1, gamma=1.0, score=0.797980 -   5.7s
[GridSearchCV] C=0.1, gamma=1.0 ................................................
[GridSearchCV] ....................... C=0.1, gamma=1.0, score=0.781145 -  26.1s
[GridSearchCV] C=0.1, gamma=10.0 ...............................................
[GridSearchCV] ...................... C=0.1, gamma=10.0, score=0.777778 -  26.9s
[GridSearchCV] C=0.1, gamma=10.0 ...............................................
[GridSearchCV] ...................... C=0.1, gamma=10.0, score=0.797980 -   5.6s
[GridSearchCV] C=0.1, gamma=10.0 ...............................................
[GridSearchCV] ...................... C=0.1, gamma=10.0, score=0.781145 -  25.4s
[GridSearchCV] C=0.1, gamma=100.0 ..............................................
[GridSearchCV] ..................... C=0.1, gamma=100.0, score=0.777778 -  27.4s
[GridSearchCV] C=0.1, gamma=100.0 ..............................................
[GridSearchCV] ..................... C=0.1, gamma=100.0, score=0.797980 -   5.7s
[GridSearchCV] C=0.1, gamma=100.0 ..............................................
[GridSearchCV] ..................... C=0.1, gamma=100.0, score=0.781145 -  26.0s
[GridSearchCV] C=1.0, gamma=0.01 ...............................................
[GridSearchCV] ...................... C=1.0, gamma=0.01, score=0.787879 -  34.6s
[GridSearchCV] C=1.0, gamma=0.01 ...............................................
[GridSearchCV] ...................... C=1.0, gamma=0.01, score=0.801347 -  28.1s[Parallel(n_jobs=1)]: Done   1 jobs       | elapsed:    0.6s
[Parallel(n_jobs=1)]: Done  32 jobs       | elapsed:  6.0min

[GridSearchCV] C=1.0, gamma=0.01 ...............................................
[GridSearchCV] ...................... C=1.0, gamma=0.01, score=0.781145 -  47.4s
[GridSearchCV] C=1.0, gamma=0.1 ................................................
[GridSearchCV] ....................... C=1.0, gamma=0.1, score=0.787879 -  34.2s
[GridSearchCV] C=1.0, gamma=0.1 ................................................
[GridSearchCV] ....................... C=1.0, gamma=0.1, score=0.801347 -  27.0s
[GridSearchCV] C=1.0, gamma=0.1 ................................................
[GridSearchCV] ....................... C=1.0, gamma=0.1, score=0.781145 -  47.2s
[GridSearchCV] C=1.0, gamma=1.0 ................................................
[GridSearchCV] ....................... C=1.0, gamma=1.0, score=0.787879 -  34.8s
[GridSearchCV] C=1.0, gamma=1.0 ................................................
[GridSearchCV] ....................... C=1.0, gamma=1.0, score=0.801347 -  27.4s
[GridSearchCV] C=1.0, gamma=1.0 ................................................
[GridSearchCV] ....................... C=1.0, gamma=1.0, score=0.781145 -  47.4s
[GridSearchCV] C=1.0, gamma=10.0 ...............................................
[GridSearchCV] ...................... C=1.0, gamma=10.0, score=0.787879 -  34.3s
[GridSearchCV] C=1.0, gamma=10.0 ...............................................
[GridSearchCV] ...................... C=1.0, gamma=10.0, score=0.801347 -  27.4s
[GridSearchCV] C=1.0, gamma=10.0 ...............................................
[GridSearchCV] ...................... C=1.0, gamma=10.0, score=0.781145 -  49.7s
[GridSearchCV] C=1.0, gamma=100.0 ..............................................
[GridSearchCV] ..................... C=1.0, gamma=100.0, score=0.787879 -  33.7s
[GridSearchCV] C=1.0, gamma=100.0 ..............................................
[GridSearchCV] ..................... C=1.0, gamma=100.0, score=0.801347 -  27.7s
[GridSearchCV] C=1.0, gamma=100.0 ..............................................
[GridSearchCV] ..................... C=1.0, gamma=100.0, score=0.781145 -  47.8s
[GridSearchCV] C=10.0, gamma=0.01 ..............................................
[GridSearchCV] ..................... C=10.0, gamma=0.01, score=0.814815 - 1.5min
[GridSearchCV] C=10.0, gamma=0.01 ..............................................
[GridSearchCV] ..................... C=10.0, gamma=0.01, score=0.794613 -  53.4s
[GridSearchCV] C=10.0, gamma=0.01 ..............................................
[GridSearchCV] ..................... C=10.0, gamma=0.01, score=0.771044 -  56.1s
[GridSearchCV] C=10.0, gamma=0.1 ...............................................
[GridSearchCV] ...................... C=10.0, gamma=0.1, score=0.814815 - 1.5min
[GridSearchCV] C=10.0, gamma=0.1 ...............................................
[GridSearchCV] ...................... C=10.0, gamma=0.1, score=0.794613 -  54.5s
[GridSearchCV] C=10.0, gamma=0.1 ...............................................
[GridSearchCV] ...................... C=10.0, gamma=0.1, score=0.771044 -  57.2s
[GridSearchCV] C=10.0, gamma=1.0 ...............................................
[GridSearchCV] ...................... C=10.0, gamma=1.0, score=0.814815 - 1.5min
[GridSearchCV] C=10.0, gamma=1.0 ...............................................
[GridSearchCV] ...................... C=10.0, gamma=1.0, score=0.794613 -  53.5s
[GridSearchCV] C=10.0, gamma=1.0 ...............................................
[GridSearchCV] ...................... C=10.0, gamma=1.0, score=0.771044 -  55.9s
[GridSearchCV] C=10.0, gamma=10.0 ..............................................
[GridSearchCV] ..................... C=10.0, gamma=10.0, score=0.814815 - 1.5min
[GridSearchCV] C=10.0, gamma=10.0 ..............................................
[GridSearchCV] ..................... C=10.0, gamma=10.0, score=0.794613 -  53.4s
[GridSearchCV] C=10.0, gamma=10.0 ..............................................
[GridSearchCV] ..................... C=10.0, gamma=10.0, score=0.771044 -  59.4s
[GridSearchCV] C=10.0, gamma=100.0 .............................................
[GridSearchCV] .................... C=10.0, gamma=100.0, score=0.814815 - 1.5min
[GridSearchCV] C=10.0, gamma=100.0 .............................................
[GridSearchCV] .................... C=10.0, gamma=100.0, score=0.794613 -  55.8s
[GridSearchCV] C=10.0, gamma=100.0 .............................................
[GridSearchCV] .................... C=10.0, gamma=100.0, score=0.771044 -  56.9s
[GridSearchCV] C=100.0, gamma=0.01 .............................................
[GridSearchCV] .................... C=100.0, gamma=0.01, score=0.804714 - 1.3min
[GridSearchCV] C=100.0, gamma=0.01 .............................................
[GridSearchCV] .................... C=100.0, gamma=0.01, score=0.808081 -  48.8s
[GridSearchCV] C=100.0, gamma=0.01 .............................................
[GridSearchCV] .................... C=100.0, gamma=0.01, score=0.771044 - 1.3min
[GridSearchCV] C=100.0, gamma=0.1 ..............................................
[GridSearchCV] ..................... C=100.0, gamma=0.1, score=0.804714 - 1.4min
[GridSearchCV] C=100.0, gamma=0.1 ..............................................
[GridSearchCV] ..................... C=100.0, gamma=0.1, score=0.808081 -  48.5s
[GridSearchCV] C=100.0, gamma=0.1 ..............................................
[GridSearchCV] ..................... C=100.0, gamma=0.1, score=0.771044 - 1.3min
[GridSearchCV] C=100.0, gamma=1.0 ..............................................
[GridSearchCV] ..................... C=100.0, gamma=1.0, score=0.804714 - 1.4min
[GridSearchCV] C=100.0, gamma=1.0 ..............................................
[GridSearchCV] ..................... C=100.0, gamma=1.0, score=0.808081 -  49.6s
[GridSearchCV] C=100.0, gamma=1.0 ..............................................
[GridSearchCV] ..................... C=100.0, gamma=1.0, score=0.771044 - 1.3min
[GridSearchCV] C=100.0, gamma=10.0 .............................................
[GridSearchCV] .................... C=100.0, gamma=10.0, score=0.804714 - 1.3min
[GridSearchCV] C=100.0, gamma=10.0 .............................................
[GridSearchCV] .................... C=100.0, gamma=10.0, score=0.808081 -  47.9s
[GridSearchCV] C=100.0, gamma=10.0 .............................................
[GridSearchCV] .................... C=100.0, gamma=10.0, score=0.771044 - 1.3min
[GridSearchCV] C=100.0, gamma=100.0 ............................................
[GridSearchCV] ................... C=100.0, gamma=100.0, score=0.804714 - 1.4min
[GridSearchCV] C=100.0, gamma=100.0 ............................................
[GridSearchCV] ................... C=100.0, gamma=100.0, score=0.808081 -  50.5s
[GridSearchCV] C=100.0, gamma=100.0 ............................................
[GridSearchCV] ................... C=100.0, gamma=100.0, score=0.771044 - 1.3min
[Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed: 48.1min finished


"""
