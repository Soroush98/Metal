import matplotlib.pyplot as plt
import pandas as pd
import numpy
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
d = pd.read_csv("tttt.csv")
d.Torsion=d.Torsion.apply([0.05 , 0.25 , 0.75].index)
X=d.loc[:,(d.columns !='Torsion')]
print(d.groupby("Temprature")["Torsion"].count())
tempreture = [1000 , 1050 , 1100]
plt.figure(1,(12,12))
# plt.scatter(x=d["Twist angle"], y=d["Torque"], marker='o', c='r', edgecolor='b')
for i in tempreture:
    indic = tempreture.index(i)
    print(indic)
    plt.subplot(1,3, indic +1 )
    new_df = d[d['Torque']== i]
    count  = new_df['Torsion'].value_counts()
    print(count)
    plt.bar([0,1,2] , count , color = ["pink" , "green" , "blue"] )
    plt.title("Count of  temprature "+ str(i))
plt.show()
print(X.shape)
# scaler = preprocessing.StandardScaler()
# scaled_values = scaler.fit_transform(X.iloc[:,:])
# X.iloc[:,:] = scaled_values
Y=d.Torsion

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print(clf.predict([[0.506,0.48169,0.299231,1000]]))
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))