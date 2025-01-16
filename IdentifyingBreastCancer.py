import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix ,accuracy_score
from sklearn.model_selection import train_test_split


cancer_data = datasets.load_breast_cancer()


features = cancer_data.data
target = cancer_data.target      #1 - Negative (WDBC-Benign)  0-Positive  (WDBC-Malignant)
train_x , test_x , train_y , test_y = train_test_split(features,target,test_size=0.3)
best_param  = {"max_depth" : np.arange(1,10)}
model = GridSearchCV(DecisionTreeClassifier(criterion="gini"),best_param)
model.fit(train_x,train_y)
predicted = model.predict(test_x)

#Result
"""
Best Parameter is :  {'max_depth': np.int64(4)}
Confusion Metric : 
 [[ 59   1]
 [  2 109]]
Accuracy Score : 
 0.9824561403508771
 """

print("Best Parameter is : ",model.best_params_)
print("Confusion Metric : \n",confusion_matrix(test_y,predicted))
print("Accuracy Score : \n",accuracy_score(test_y,predicted))
