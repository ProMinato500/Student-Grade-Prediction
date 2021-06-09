import pandas as pd
import numpy as np

data=pd.read_csv("/content/student-mat.csv")

data.head()

data['G3'].describe()

import seaborn as sns
import matplotlib.pyplot as plt
demo= sns.countplot(data['G3'])
demo.axes.set_title('Distribution of Final grade of students', fontsize = 35)
demo.set_xlabel('Final Grade', fontsize = 20)
demo.set_ylabel('Count', fontsize = 20)
plt.show()

data.isnull().any()

data.isnull().sum()

male_student = len(data[data['sex'] == 'M'])
female_student= len(data[data['sex'] == 'F'])
print('Number of male students:',male_student)
print('Number of female students:',female_student)

demo= sns.countplot('age',hue='sex', data=data)
demo.axes.set_title('age groups',fontsize=30)
demo.set_xlabel("Age",fontsize=30)
demo.set_ylabel("Count",fontsize=20)
plt.show()

demo = sns.countplot(data['address'])
demo.axes.set_title('Urban Vs rural students', fontsize = 30)
demo.set_xlabel('Address', fontsize = 20)
demo.set_ylabel('Count', fontsize = 20)
plt.show()

data.corr()['G3'].sort_values()

data['GradeAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3

data.drop(["school","age"], axis=1, inplace=True)

data.head()

data_dum=data

categorical_d = {'yes': 1, 'no': 0}
data_dum['schoolsup'] = data_dum['schoolsup'].map(categorical_d)
data_dum['famsup'] = data_dum['famsup'].map(categorical_d)
data_dum['paid'] = data_dum['paid'].map(categorical_d)
data_dum['activities'] = data_dum['activities'].map(categorical_d)
data_dum['nursery'] = data_dum['nursery'].map(categorical_d)
data_dum['higher'] = data_dum['higher'].map(categorical_d)
data_dum['internet'] = data_dum['internet'].map(categorical_d)
data_dum['romantic'] = data_dum['romantic'].map(categorical_d)

categorical_d = {'F': 1, 'M': 0}
data_dum['sex'] = data_dum['sex'].map(categorical_d)


categorical_d = {'U': 1, 'R': 0}
data_dum['address'] = data_dum['address'].map(categorical_d)


categorical_d = {'LE3': 1, 'GT3': 0}
data_dum['famsize'] = data_dum['famsize'].map(categorical_d)


categorical_d= {'T': 1, 'A': 0}
data_dum['Pstatus'] = data_dum['Pstatus'].map(categorical_d)


categorical_d = {'teacher': 0, 'health': 1, 'services': 2,'at_home': 3,'other': 4}
data_dum['Mjob'] = data_dum['Mjob'].map(categorical_d)
data_dum['Fjob'] = data_dum['Fjob'].map(categorical_d)


categorical_d= {'home': 0, 'reputation': 1, 'course': 2,'other': 3}
data_dum['reason'] = data_dum['reason'].map(categorical_d)


categorical_d = {'mother': 0, 'father': 1, 'other': 2}
data_dum['guardian'] = data_dum['guardian'].map(categorical_d)

data_dum.columns

from sklearn.model_selection import train_test_split
x=data_dum.drop("G3",axis=1)
y=data_dum['G3']

data_dum['G3']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=44)

from sklearn.linear_model import LinearRegression

L=LinearRegression()

L.fit(X_train, y_train)

y_pred=L.predict(X_test)

print(L.score(X_test, y_test))