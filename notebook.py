import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

# Feature Engineering - Best Version
for df in [train, test]:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)

# Fill Age by Title - This is the SECRET for 80+
for df in [train, test]:
    for title in df['Title'].unique():
        df.loc[(df['Age'].isna()) & (df['Title']==title), 'Age'] = df[df['Title']==title]['Age'].median()

train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())

for df in [train, test]:
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2})
    df['Title'] = df['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}).fillna(3)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['AgeClass'] = df['Age'] * df['Pclass']

features = ['Pclass','Sex','Age','Fare','Embarked','Title','FamilySize','IsAlone','HasCabin','FarePerPerson','AgeClass']

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(train[features], train['Survived'])

pred = model.predict(test[features])

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})
submission.to_csv('submission.csv', index=False)
print("DONE! This will be 0.80+")
DONE! This will be 0.80+
