import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

from ggplot import ggplot, aes, geom_bar, geom_histogram

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

    # Remove the columns that should be dropped
    df = df.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis=1) 
    
    # Missing Fares should be median of their respective classes
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().median()
    for f in range(0,3):
        df.loc[df.Fare.isnull() & df.Pclass == f+1, 'Fare'] = median_fare[f]
    
    return(df)

# Load the training data.
train_df = pd.read_csv('train-original.csv', header=0)

# Convert Embarked to int
# Breaking this down:
# np.unique returns unique values in the input 
# enumerate gives you an object that can give you tuples of index and value
ports = enumerate(np.unique(train_df['Embarked']))
ports_dict = { name: i for i, name in ports }

train_df = clean_up(train_df, ports_dict)

# Test Data
test_df = pd.read_csv('test.csv', header=0)  
ids = test_df['PassengerId'].values
test_df = clean_up(test_df, ports_dict)

### Random Forest
# Training
# sklearn needs numpy arrays, not DataFrames, so convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

#print(train_df.info())
#print(test_df.info())

forest = RandomForestClassifier(n_estimators=500)
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

scores = cross_validation.cross_val_score(forest, train_data[0::,1::], train_data[0::,0], cv=5)

# >>> clf = svm.SVC(kernel='linear', C=1)
# >>> scores = cross_validation.cross_val_score(
# ...    clf, iris.data, iris.target, cv=5)

# Calculate training error
#score = forest.score(train_data[0::,1::], train_data[0::,0])
#print(score * 100)

#print(forest.feature_importances_)
#print(train_df.info())

output = forest.predict(test_data[0::,1::]).astype(int)

predictions_file = open("forestPython.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

### LogisticRegression
logistic = LogisticRegression()
logistic = logistic.fit(train_data[0::,1::], train_data[0::,0])

# Calculate training error
score = logistic.score(train_data[0::,1::], train_data[0::,0])
print(score * 100)

output = logistic.predict(test_data[0::,1::]).astype(int)

predictions_file = open("logistic.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()





