import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from ggplot import ggplot, aes, geom_bar, geom_histogram

import random
import re

def replace_titles(data):
    title = data['Title']
    
    if title in ['Don', 'Capt', 'Major']:
        return 'Sir'
    elif title in ['Dona', 'the Countess', 'Jonkheer']:
        return 'Lady'
    else:
        return title

def clean_up(df, ports_dict):
    # Clean up the data.
    # Columns with nulls: Age, Embarked
    # Columns that need to be converted to ints: Sex, Embarked
    # Columns that should be dropped: Name, PassengerId, Cabin, Ticket

    # Make NA ages the median of all ages
    # Possible improvement: Use median age by gender
    male_median_age = df.loc[df.Sex == 'male', 'Age'].dropna().median()
    female_median_age = df.loc[df.Sex == 'female', 'Age'].dropna().median()
    
    df.loc[(df['Sex'] == 'male') & (df['Age'].isnull()), 'Age'] = male_median_age
    df.loc[(df['Sex'] == 'female') & (df['Age'].isnull()), 'Age'] = female_median_age
    
    # Make NA Embarked be the mode.
    # Possible improvement: Randomly sample using the same distribution of ports
    df.loc[df.Embarked.isnull(), 'Embarked'] = df.Embarked.dropna().mode().values
    
    df.Embarked = df.Embarked.map(lambda x: ports_dict[x]).astype(int)
    
    # Convert Sex to integer
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Extract titles
    df['Title'] = [re.split(', |\. ', name)[1] for name in df.Name]
    df['Title'] = df.apply(replace_titles, axis = 1)
    
    # Use dummies to generate a column for each of the 12 remaining columns (have to use all 12 here if possible)

    # Missing Fares should be median of their respective classes
    median_fare = np.zeros(3)
    for f in range(0, 3):
        median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().median()
    for f in range(0, 3):
        df.loc[df.Fare.isnull() & df.Pclass == f+1, 'Fare'] = median_fare[f]
    
    titles = pd.get_dummies(combo_df['Title'])
    df = pd.concat([df, titles], axis = 1)
    
    # Create FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Remove the columns that should be dropped
    df = df.drop(['Name', 'PassengerId', 'Cabin', 'Ticket', 'Title'], axis=1) 
    
    return(df)

train_df = pd.read_csv('trainl.csv', header=0)
test_df = pd.read_csv('test.csv', header=0) 

ids = test_df['PassengerId']

combo_df = train_df.append(test_df)
# Convert Embarked to int
# Breaking this down:
# np.unique returns unique values in the input 
# enumerate gives you an object that can give you tuples of index and value
ports = enumerate(np.unique(train_df['Embarked']))
ports_dict = { name: i for i, name in ports }

combo_df = clean_up(combo_df, ports_dict)

# Move Survived into the first column
column_list = combo_df.columns.tolist()
column_list.remove('Survived')
column_list = ['Survived'] + column_list
combo_df = combo_df[column_list]

train_df = combo_df[0:891]
test_df = combo_df[891::]

### Random Forest
# Training
# sklearn needs numpy arrays, not DataFrames, so convert back to a numpy array

np.random.seed(1)
forest = RandomForestClassifier(n_estimators=500)
#forest = forest.fit(train_data[0::,1::], train_data[0::,0])

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# set up parameters
parameters = {'n_estimators': [4, 6, 10, 50, 500], 
              'max_features': ['log2', 'sqrt', 'auto'], 
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# grid search CV
grid_obj = GridSearchCV(forest, parameters, verbose=1, n_jobs=2)
grid_obj = grid_obj.fit(train_df[predictors], train_df['Survived'])

bestForest = grid_obj.best_estimator_
bestForest.fit(train_df[predictors], train_df['Survived'])

output = bestForest.predict(test_df[predictors]).astype(int)

predictions_file = open("forestPython.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

### LogisticRegression
logistic = LogisticRegression()
logistic = logistic.fit(train_df[predictors], train_df["Survived"])

logistic.get_params()

# Calculate training error
score = logistic.score(train_df[predictors], train_df["Survived"])
print("Logistic accuracy: %.2f" % (score * 100))


output = logistic.predict(test_df[predictors]).astype(int)

predictions_file = open("logisticPython.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()





























