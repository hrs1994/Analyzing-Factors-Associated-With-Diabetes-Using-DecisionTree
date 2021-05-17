# importing the libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from graphviz import Source
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# importing the dataset
df = pd.read_csv('diabetes.csv')
# creating the tree

variables = list(df.columns[:8])
y = df['Outcome']
X = df[variables]
Tree = tree.DecisionTreeClassifier(max_depth=3)
Tree = Tree.fit(X, y)
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(iris.data, iris.target)
dot_data_exp = tree.export_graphviz(Tree, out_file = None, feature_names = X.columns, class_names= ['0','1'], filled = True, rounded = True, special_characters = True)
# visualizing the tree
graph = Source(dot_data_exp)
graph.render('diabetes')
graph.view()
# evaluating the model
# training tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
bc_tree = tree.DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)
# calculating the prdiction
bc_pred = bc_tree.predict(X_test)
# evaluting the scores
bc_tree.score(X_test, y_test)
# creating the confusion/error matrix
accuracy = accuracy_score(y_test,bc_pred)
report = classification_report(y_test,bc_pred)
print(accuracy)
print(report)
cm = confusion_matrix(y_test, bc_pred)
print(cm)
# visualizing the error matrix
plt.imshow(cm, cmap = 'binary', interpolation = 'None')
'''fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels(['a'])
ax.set_yticklabels(['b'])
plt.xlabel('Predicted')
plt.ylabel('True')'''
plt.show()
