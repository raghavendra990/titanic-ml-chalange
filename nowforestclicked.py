import pandas as pd 
import numpy as np 
import  csv as csv 
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier

'''
train = pd.read_csv('train.csv')
#print train.head(5)

survived_column = train['Survived']
survived_column = survived_column.values

numerical_features = train[['Fare','Pclass','Age']]

median_features = numerical_features.dropna().median()

imputed_features = numerical_features.fillna(median_features)
features_array = imputed_features.values

rich_features = pd.concat([train[['Fare', 'Pclass', 'Age']],
                           pd.get_dummies(train['Sex'], prefix='Sex'),
                           pd.get_dummies(train['Embarked'], prefix='Embarked')],
                          axis=1)

rich_features_no_male = rich_features.drop('Sex_male',1)

rich_features_final = rich_features_no_male.fillna(rich_features_no_male.dropna().median())


test = pd.read_csv('test.csv')

t_numerical_features = test[['Fare','Pclass','Age']]

t_median_features = t_numerical_features.dropna().median()

t_imputed_features = t_numerical_features.fillna(t_median_features)
t_features_array = t_imputed_features.values

t_rich_features = pd.concat([test[['Fare', 'Pclass', 'Age']],
                           pd.get_dummies(test['Sex'], prefix='Sex'),
                           pd.get_dummies(test['Embarked'], prefix='Embarked')],
                          axis=1)

t_rich_features_no_male = t_rich_features.drop('Sex_male',1)
t_rich_features_final = t_rich_features_no_male.fillna(t_rich_features_no_male.dropna().median())

model = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, random_state=50, max_features = "auto", min_samples_leaf = 50)

model.fit(rich_features_final, survived_column)

output = model.predict(t_rich_features_final).astype(int)
ids = test['PassengerId'].values
predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
x = pd.read_csv("train.csv")

y = x.pop("Survived")

model =  RandomForestRegressor(n_estimator = 100 , oob_score = TRUE, random_state = 42)

model.fit(x(numeric_variable,y))

print "AUC - ROC : ", roc_auc_score(y,model.oob_prediction)

