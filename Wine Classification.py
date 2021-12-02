

# Import/Set-up (Run before Analysis) #

# Import #

# Import Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# Import Kaggle Dataset
# Data Link: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

wine_data = pd.read_csv('winequality-red.csv')
print("Null values:", wine_data.isna().sum())
wine_data.dropna(inplace = True)

"""### ***Target Variable Creation (1=Good, 0=Bad)***"""

# 1 = Good, 0 = Bad
wine_data['binary quality'] = np.where(wine_data['quality'] >= 6, 1, 0)

# Drop Original Quality Ratings
wine_data.drop(['quality'], axis = 1, inplace = True)

"""### ***Exploratory Analysis***"""

print(wine_data.head())

# Binary Quality Value Counts
print('Binary Quality Value Counts:\n', wine_data['binary quality'].value_counts())

"""### ***Train/Test Split***"""

# Prediction Variables
X = wine_data.drop(['binary quality'], axis = 1)

# Target Variable
y = wine_data['binary quality']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

"""## ***70:30 Holdout Validation***

### ***Decision Tree Classification Model ***

***Decision Tree Classification***

***GINI Index***
"""

# Train Data
gini_decision_tree = DecisionTreeClassifier()
gini_decision_tree = gini_decision_tree.fit(X_train, y_train)

# Test Data
gini_y_pred = gini_decision_tree.predict(X_test)

"""***Entropy***"""

# Train Data
entropy_decision_tree = DecisionTreeClassifier(criterion = 'entropy')
entropy_decision_tree = entropy_decision_tree.fit(X_train, y_train)

# Test Data
entropy_y_pred = entropy_decision_tree.predict(X_test)

"""***Optimal Decision Tree***"""

# Train Data
optimal_decision_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 13)
optimal_decision_tree = optimal_decision_tree.fit(X_train, y_train)

# Test Data
optimal_y_pred = optimal_decision_tree.predict(X_test)

"""***Decision Tree Optimization***

***GINI vs. Entropy***
"""

# GINI Accuracy #
print("GINI Accuracy:", metrics.accuracy_score(y_test,gini_y_pred))
print("GINI Precision:",metrics.precision_score(y_test, gini_y_pred, pos_label = 1))
print("GINI Recall:",metrics.recall_score(y_test, gini_y_pred, pos_label = 1))
print("GINI F-1 score:",metrics.f1_score(y_test, gini_y_pred, pos_label = 1))

# Entropy Accuracy #
print("Entropy Accuracy:", metrics.accuracy_score(y_test,entropy_y_pred))
print("Entropy Precision:",metrics.precision_score(y_test, entropy_y_pred, pos_label = 1))
print("Entropy Recall:",metrics.recall_score(y_test, entropy_y_pred, pos_label = 1))
print("Entropy F-1 score:",metrics.f1_score(y_test, entropy_y_pred, pos_label = 1))

# Conclusion: Choose GINI because it has a larger F-1 score

"""***GINI Node Amounts***"""

max_depth_range = list(range(1, 20))

for depth in max_depth_range:
    
    node_decision_tree = DecisionTreeClassifier(max_depth = depth, 
                             random_state = 0)
    node_decision_tree.fit(X_train, y_train)
    node_y_pred = node_decision_tree.predict(X_test)
    score = metrics.f1_score(y_test, node_y_pred, pos_label = 1)
    print('Number of Nodes:', depth, 'F-1 Score:', score)

# Conclusion: Choose 13 Nodes because it has a larger F-1 score

"""***Decision Tree Performance Analysis***

***Optimal Classification Tree Performance Analysis***
"""

# Accuracy #
class_tree_optimal_accuracy = metrics.accuracy_score(y_test,optimal_y_pred)
print("Optimal Classification Tree Accuracy:", class_tree_optimal_accuracy)

# Precision #
class_tree_optimal_precision = metrics.precision_score(y_test, optimal_y_pred, pos_label = 1)
print("Optimal Classification Tree Precision:", class_tree_optimal_precision)

# Recall #
class_tree_optimal_recall = metrics.recall_score(y_test, optimal_y_pred, pos_label = 1)
print("Optimal Classification Tree Recall:", class_tree_optimal_recall)

# F-1 Score #
class_tree_optimal_f1score = metrics.f1_score(y_test, optimal_y_pred, pos_label = 1)
print("Optimal Classification Tree F-1 score:",class_tree_optimal_f1score)

# Confusion Matrix #
class_tree_optimal_confusion_matrix = metrics.confusion_matrix(y_test, optimal_y_pred)
print("Optimal Classification Tree Confusion Matrix:\n", class_tree_optimal_confusion_matrix)

# ROC #
class_tree_optimal_roc = metrics.roc_curve(y_test, optimal_y_pred, pos_label = 1)
print("Optimal Classification Tree ROC:", class_tree_optimal_roc)

# ROC/AUC Score#
class_tree_optimal_rocauc = metrics.roc_auc_score(y_test, optimal_y_pred)
print("Optimal Classification Tree ROC AUC Score:", class_tree_optimal_rocauc)

"""***Optimal Classification Tree Visualization***"""

# Visualization of Best Decision Tree Model
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

dot_data = StringIO()
export_graphviz(optimal_decision_tree, out_file=dot_data,filled=True, rounded=True, special_characters=True,
                feature_names = feature_cols,class_names=['Bad','Good'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('wine_data_class_tree.png')
Image(graph.create_png())

"""### ***Logistic Regression Classification Model ***

***Logistic Regression Classification***
"""

#import the class for logistic regression 
from sklearn.linear_model import LogisticRegression

#instantiate the model (using the default parameters)
logreg = LogisticRegression()

#fit the model with data
logreg.fit(X_train, y_train)

#
y_pred=logreg.predict(X_test)

#compute classification performance metrics 
from sklearn import metrics
from sklearn.metrics import roc_auc_score

print('accuracy_score')
print(metrics.accuracy_score(y_test, y_pred))
print('precision')
print(metrics.precision_score(y_test, y_pred))
print('recall_score')
print(metrics.recall_score(y_test, y_pred))
print('f1_score')
print(metrics.f1_score(y_test, y_pred))
print('log_loss')
print(metrics.log_loss(y_test, y_pred))
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC)')
print(roc_auc_score(y_test, y_pred))

#get the confusion matrix
from sklearn.metrics import confusion_matrix
print('confusion matrix')
print(confusion_matrix(y_test, y_pred))

#print odds ratios
# Pandas and NumPy import
import numpy as np

print(feature_cols)
np.exp(logreg.coef_)

"""***Logistic Regression Performance Analysis***"""

# Visual
import seaborn as sns
import matplotlib.pyplot as plt
labels = ['Bad', 'Good']

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, square = True, linewidths=.5, xticklabels= labels, yticklabels= labels, fmt = 'g')

plt.xlabel('Predicted', fontsize = 15)
plt.ylabel('Actual', fontsize = 15)
plt.show()

# Visual Important Features
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
feature_imp = pd.Series(logreg.coef_[0], index = feature_cols).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)

# Adding Labels to bar plot
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


"""### ***Naive Bayes Classification Model ***

***Naive Bayes Classification***
"""

# Train Data
naive_bayes = GaussianNB()
naive_bayes = naive_bayes.fit(X_train, y_train)

# Test Data
nb_y_pred = naive_bayes.predict(X_test)

"""***Naive Bayes Performance Analysis***"""

# Accuracy #
naive_bayes_accuracy = metrics.accuracy_score(y_test, nb_y_pred)
print("Naive Bayes Accuracy:", naive_bayes_accuracy)

# Precision #
naive_bayes_precision = metrics.precision_score(y_test, nb_y_pred, pos_label = 1)
print("Naive Bayes Precision:", naive_bayes_precision)

# Recall #
naive_bayes_recall = metrics.recall_score(y_test, nb_y_pred, pos_label = 1)
print("Naive Bayes Recall:", naive_bayes_recall)

# F-1 Score #
naive_bayes_f1score = metrics.f1_score(y_test, nb_y_pred, pos_label = 1)
print("Naive Bayes F-1 score:", naive_bayes_f1score)

# Confusion Matrix #
naive_bayes_confusion_matrix = metrics.confusion_matrix(y_test, nb_y_pred)
print("Naive Bayes Confusion Matrix:\n", naive_bayes_confusion_matrix)

# ROC #
naive_bayes_roc = metrics.roc_curve(y_test, nb_y_pred, pos_label = 1)
print("Naive Bayes ROC:", naive_bayes_roc)

# ROC/AUC Score #
naive_bayes_rocauc = metrics.roc_auc_score(y_test, nb_y_pred)
print("Naive Bayes ROC AUC Score:", naive_bayes_rocauc)

"""### ***Support Machine Vector Classification Model ***

***Support Machine Vector Classification***
"""

#SVM Classifier 
clf = svm.SVC(kernel='linear')

#Train the model using training sets 
clf.fit(X_train,y_train)

#Predict the response for the test dataset 
y_pred = clf.predict(X_test)

"""***Support Machine Vector Performance Analysis***"""

# Accuracy: 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Precision: 
print("Precision:",metrics.precision_score(y_test, y_pred))

# Recall: 
print("Recall:",metrics.recall_score(y_test, y_pred))

# F1 Score
print('F1 Score:', metrics.f1_score(y_test, y_pred))

"""### ***Neural Network Classification Model***"""

# Train Data
neural_network = MLPClassifier(hidden_layer_sizes= (3,2), random_state=1)
neural_network = neural_network.fit(X_train, y_train)

# Test
neural_network_y_pred = neural_network.predict(X_test)

# Accuracy Metrics
print("Neural Network Accuracy:", metrics.accuracy_score(y_test,neural_network_y_pred))
print("Neural Network Precision:",metrics.precision_score(y_test, neural_network_y_pred))
print("Neural Network Recall:",metrics.recall_score(y_test, neural_network_y_pred))
print("Neural Network F-1 score:",metrics.f1_score(y_test, neural_network_y_pred))

"""### ***Random Forest Classification Model***"""

clf=RandomForestClassifier(n_estimators=100)

#Train the model & predict
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Random Forest Precision:",metrics.precision_score(y_test, y_pred))
print("Random Forest Recall:",metrics.recall_score(y_test, y_pred))
print("Random Forest F-1 Score:",metrics.f1_score(y_test, y_pred))

# Load dataset
wine = datasets.load_wine()

# the target names
target_name = wine.target_names
print(target_name)

# the names of the features
feature_names = list(X.columns)
print(feature_names)

# Finding Important Features
feature_imp = pd.Series(clf.feature_importances_, index = feature_names).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)

# Adding Labels to bar plot
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# Creating Visual
clf_visual=RandomForestClassifier(n_estimators=100, max_depth = 3)

#Train the model & predict
clf_visual.fit(X_train,y_train)
y_pred=clf_visual.predict(X_test)

from sklearn.tree import export_graphviz
estimator = clf_visual.estimators_[5]
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

export_graphviz(estimator, out_file = 'tree.dot', feature_names = feature_cols, class_names = ['Bad','Good'], rounded = True, proportion = False, precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')

# Prediction Variables
X_dropliv = wine_data.drop(['binary quality','free sulfur dioxide'], axis = 1) # dropping lowest importance variable-- "free sulfur dioxide"

# Target Variable
y_dropliv = wine_data['binary quality']

# Split Data
X_train_dropliv, X_test_dropliv, y_train_dropliv, y_test_dropliv = train_test_split(X_dropliv, y_dropliv, test_size=0.3, random_state=1)

# Performance Metrics sans "free sulfur dioxide" variable
clf_dropliv=RandomForestClassifier(n_estimators=100)

#Train model 
clf_dropliv.fit(X_train_dropliv,y_train_dropliv)

# prediction on test set
y_pred_dropliv=clf_dropliv.predict(X_test_dropliv)

# Performance Metrics
print("Random Forest (without least important feature) Accuracy:",metrics.accuracy_score(y_test_dropliv, y_pred_dropliv))
print("Random Forest (without least important feature) Precision:",metrics.precision_score(y_test_dropliv, y_pred_dropliv))
print("Random Forest (without least important feature) Recall:",metrics.recall_score(y_test_dropliv, y_pred_dropliv))
print("Random Forest (without least important feature) F-1 Score:",metrics.f1_score(y_test_dropliv, y_pred_dropliv))

"""### ***Bagging Ensemble Classification Model***"""

# Train Data
bagging_ensemble = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, random_state=0)
bagging_ensemble = bagging_ensemble.fit(X_train, y_train) 

# Test Data
bagging_ensemble_y_pred = bagging_ensemble.predict(X_test)

# Accuracy Metrics
print("Bagging Ensemble Accuracy:", metrics.accuracy_score(y_test,bagging_ensemble_y_pred))
print("Bagging Ensemble Precision:",metrics.precision_score(y_test, bagging_ensemble_y_pred))
print("Bagging Ensemble Recall:",metrics.recall_score(y_test, bagging_ensemble_y_pred))
print("Bagging Ensemble F-1 score:",metrics.f1_score(y_test, bagging_ensemble_y_pred))

"""### ***Boosting Ensemble Classification Model***"""

# Train Data
boosting_ensemble = AdaBoostClassifier()
boosting_ensemble = boosting_ensemble.fit(X_train, y_train)

# Test Data
boosting_ensemble_y_pred = boosting_ensemble.predict(X_test)

# Accuracy Metrics
print("Boosting Ensemble Accuracy:", metrics.accuracy_score(y_test,boosting_ensemble_y_pred))
print("Boosting Ensemble Precision:",metrics.precision_score(y_test, boosting_ensemble_y_pred))
print("Boosting Ensemble Recall:",metrics.recall_score(y_test, boosting_ensemble_y_pred))
print("Boosting Ensemble F-1 score:",metrics.f1_score(y_test, boosting_ensemble_y_pred))

"""### ***Gradient Tree Boosting Model***"""

# Train Data
gradient_boosting = GradientBoostingClassifier()
gradient_boosting = gradient_boosting.fit(X_train, y_train)

# Test Data
gradient_boosting_y_pred = gradient_boosting.predict(X_test)

# Accuracy Metrics
print("Gradient Tree Boosting Accuracy:", metrics.accuracy_score(y_test,gradient_boosting_y_pred))
print("Gradient Tree Boosting Precision:",metrics.precision_score(y_test, gradient_boosting_y_pred))
print("Gradient Tree Boosting Recall:",metrics.recall_score(y_test, gradient_boosting_y_pred))
print("Gradient Tree Boosting Accuracy F-1 score:",metrics.f1_score(y_test, gradient_boosting_y_pred))

"""### ***Voting Ensemble Method***"""

# Voting Ensemble for 3 Best Classification Models (Run selected models first!)
voting_ensemble = VotingClassifier(estimators = [('Classification Tree', optimal_decision_tree), ('Boosting Ensemble', boosting_ensemble), ('Naive Bayes', naive_bayes)])

# Accuracy
for clf, label in zip([optimal_decision_tree, boosting_ensemble, naive_bayes, voting_ensemble], ['Classification Tree', 'Boosting Ensemble', 'Naive Bayes', 'Voting Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='f1', cv=5)
    print("F-1 Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

"""### ***K Nearest Neighbors***"""

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)

# Accuracy # 
print('KNN Accuracy:', metrics.accuracy_score(y, y_pred))

# Precision #
print('KNN Precision:', metrics.precision_score(y, y_pred, pos_label=1))

# Recall #
print('KNN Recall:', metrics.recall_score(y, y_pred, pos_label=1))

# F-1 Score #
print('KNN F-1 Score:', metrics.f1_score(y, y_pred, pos_label=1))

# Confusion Matrix #
print('KNN Confusion Matrix:', metrics.confusion_matrix(y, y_pred))

# ROC #
print('KNN ROC:', metrics.roc_curve(y, y_pred, pos_label=1))

# ROC/AUC Score #
print('KNN ROC AUC Score:', metrics.roc_auc_score(y, y_pred))

"""## ***10-fold Cross Validation***

### ***Decision Tree Classification Model ***
"""

# Train Data
optimal_decision_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 13)

# Performace Metrics

# Accuracy #
tree_cross_accuracy = cross_val_score(optimal_decision_tree, X, y, scoring = 'accuracy', cv = 10)
print("Decision Tree Accuracy w/ 10-fold Cross-validation:", tree_cross_accuracy.mean())

# Precision #
tree_cross_precision = cross_val_score(optimal_decision_tree, X, y, scoring = 'precision', cv = 10)
print("Decision Tree Precision w/ 10-fold Cross-validation:",tree_cross_precision.mean())

# Recall #
tree_cross_recall = cross_val_score(optimal_decision_tree, X, y, scoring = 'recall', cv = 10)
print("Decision Tree Recall w/ 10-fold Cross-validation:",tree_cross_recall.mean())

# F-1 Score #
tree_cross_f1 = cross_val_score(optimal_decision_tree, X, y, scoring = 'f1', cv = 10)
print("Decision Tree F1 w/ 10-fold Cross-validation:",tree_cross_f1.mean())

# ROC/AUC Score#
tree_cross_rocauc = cross_val_score(optimal_decision_tree, X, y, scoring = 'roc_auc', cv = 10)
print("Decision Tree ROC AUC w/ 10-fold Cross-validation:",tree_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(optimal_decision_tree, X, y, cv=10)
tree_cross_confusion = confusion_matrix(y, y_pred)
print("Optimal Classification Tree Confusion Matrix:\n", tree_cross_confusion)

"""### ***Logistic Regression Classification Model ***"""

logreg = LogisticRegression()

# Performace Metrics

# Accuracy #
logreg_cross_accuracy = cross_val_score(logreg, X, y, scoring = 'accuracy', cv = 10)
print("Logistic Regression Accuracy w/ 10-fold Cross-validation:", logreg_cross_accuracy.mean())

# Precision #
logreg_cross_precision = cross_val_score(logreg, X, y, scoring = 'precision', cv = 10)
print("Logistic Regression Precision w/ 10-fold Cross-validation:",logreg_cross_precision.mean())

# Recall #
logreg_cross_recall = cross_val_score(logreg, X, y, scoring = 'recall', cv = 10)
print("Logistic Regression Recall w/ 10-fold Cross-validation:",logreg_cross_recall.mean())

# F-1 Score #
logreg_cross_f1 = cross_val_score(logreg, X, y, scoring = 'f1', cv = 10)
print("Logistic Regression F1 w/ 10-fold Cross-validation:",logreg_cross_f1.mean())

# ROC/AUC Score#
logreg_cross_rocauc = cross_val_score(logreg, X, y, scoring = 'roc_auc', cv = 10)
print("Logistic Regression ROC AUC w/ 10-fold Cross-validation:",logreg_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(logreg, X, y, cv=10)
logreg_cross_confusion = confusion_matrix(y, y_pred)
print("Logistic Regression Confusion Matrix:\n", logreg_cross_confusion)


"""### ***Naive Bayes Classification Model ***"""

# Train Data
naive_bayes = GaussianNB()

# Performace Metrics

# Accuracy #
naive_bayes_cross_accuracy = cross_val_score(naive_bayes, X, y, scoring = 'accuracy', cv = 10)
print("Naive Bayes Accuracy w/ 10-fold Cross-validation:", naive_bayes_cross_accuracy.mean())

# Precision #
naive_bayes_cross_precision = cross_val_score(naive_bayes, X, y, scoring = 'precision', cv = 10)
print("Naive Bayes Precision w/ 10-fold Cross-validation:",naive_bayes_cross_precision.mean())

# Recall #
naive_bayes_cross_recall = cross_val_score(naive_bayes, X, y, scoring = 'recall', cv = 10)
print("Naive Bayes Recall w/ 10-fold Cross-validation:",naive_bayes_cross_recall.mean())

# F-1 Score #
naive_bayes_cross_f1 = cross_val_score(naive_bayes, X, y, scoring = 'f1', cv = 10)
print("Naive Bayes F1 w/ 10-fold Cross-validation:",naive_bayes_cross_f1.mean())

# ROC/AUC Score#
naive_bayes_cross_rocauc = cross_val_score(naive_bayes, X, y, scoring = 'roc_auc', cv = 10)
print("Naive Bayes ROC AUC w/ 10-fold Cross-validation:",naive_bayes_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(naive_bayes, X, y, cv=10)
naive_bayes_cross_confusion = confusion_matrix(y, y_pred)
print("Naive Bayes Confusion Matrix:\n", naive_bayes_cross_confusion)

"""### ***Support Machine Vector Classification Model ***"""

#SVM Classifier 
clf = svm.SVC(kernel='linear',C = 1)

#Train the model using training sets 
clf.fit(X_train,y_train)

#Predict the response for the test dataset 
y_pred = clf.predict(X_test)

# Model Accuracy: 
support_vector_machine_cross_accuracy = cross_val_score(clf,X,y,scoring= 'accuracy', cv = 10)
print("SVM Accuracy w/ 10-fold Cross-validation:",support_vector_machine_cross_accuracy.mean())

# Model Precision: 
support_vector_machine_cross_precision = cross_val_score(clf,X,y,scoring= 'precision', cv = 10)
print("SVM Precision w/ 10-fold Cross-validation:",support_vector_machine_cross_precision.mean())

# Model Recall: 
support_vector_machine_cross_recall = cross_val_score(clf,X,y,scoring= 'recall', cv = 10)
print("SVM Recall w/ 10-fold Cross-validation:",support_vector_machine_cross_precision.mean())

# Model F1: 
support_vector_machine_cross_f1 = cross_val_score(clf,X,y,scoring= 'f1', cv = 10)
print("SVM F1 w/ 10-fold Cross-validation:",support_vector_machine_cross_f1.mean())

"""### ***Neural Network Classification Model***"""

# Train Data
neural_network = MLPClassifier(hidden_layer_sizes= (3,2), random_state=1)

# Performace Metrics

# Accuracy #
neural_network_cross_accuracy = cross_val_score(neural_network, X, y, scoring = 'accuracy', cv = 10)
print("Naive Bayes Accuracy w/ 10-fold Cross-validation:", neural_network_cross_accuracy.mean())

# Precision #
neural_network_cross_precision = cross_val_score(neural_network, X, y, scoring = 'precision', cv = 10)
print("Naive Bayes Precision w/ 10-fold Cross-validation:",neural_network_cross_precision.mean())

# Recall #
neural_network_cross_recall = cross_val_score(neural_network, X, y, scoring = 'recall', cv = 10)
print("Naive Bayes Recall w/ 10-fold Cross-validation:",neural_network_cross_recall.mean())

# F-1 Score #
neural_network_cross_f1 = cross_val_score(neural_network, X, y, scoring = 'f1', cv = 10)
print("Naive Bayes F1 w/ 10-fold Cross-validation:",neural_network_cross_f1.mean())

# ROC/AUC Score#
neural_network_cross_rocauc = cross_val_score(neural_network, X, y, scoring = 'roc_auc', cv = 10)
print("Naive Bayes ROC AUC w/ 10-fold Cross-validation:",neural_network_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(neural_network, X, y, cv=10)
neural_network_cross_confusion = confusion_matrix(y, y_pred)
print("Naive Bayes Confusion Matrix:\n", neural_network_cross_confusion)

"""### ***Random Forest Classification Model***"""

# Train Data
clf=RandomForestClassifier(n_estimators=100)

# Performace Metrics

# Accuracy #
clf_cross_accuracy = cross_val_score(clf, X, y, scoring = 'accuracy', cv = 10)
print("Random Forest Accuracy w/ 10-fold Cross-validation:", clf_cross_accuracy.mean())

# Precision #
clf_cross_precision = cross_val_score(clf, X, y, scoring = 'precision', cv = 10)
print("Random Forest Precision w/ 10-fold Cross-validation:",clf_cross_precision.mean())

# Recall #
clf_cross_recall = cross_val_score(clf, X, y, scoring = 'recall', cv = 10)
print("Random Forest Recall w/ 10-fold Cross-validation:",clf_cross_recall.mean())

# F-1 Score #
clf_cross_f1 = cross_val_score(clf, X, y, scoring = 'f1', cv = 10)
print("Random Forest F1 w/ 10-fold Cross-validation:",clf_cross_f1.mean())

# ROC/AUC Score#
clf_cross_rocauc = cross_val_score(clf, X, y, scoring = 'roc_auc', cv = 10)
print("Random Forest ROC AUC w/ 10-fold Cross-validation:",clf_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(clf, X, y, cv=10)
clf_cross_confusion = confusion_matrix(y, y_pred)
print("Random Forest Confusion Matrix:\n", clf_cross_confusion)

# sans "free sulfur dioxide" variable

# Performace Metrics

# Accuracy #
clf_dropliv_cross_accuracy = cross_val_score(clf_dropliv, X_dropliv, y_dropliv, scoring = 'accuracy', cv = 10)
print("Random Forest Accuracy w/ 10-fold Cross-validation:", clf_dropliv_cross_accuracy.mean())

# Precision #
clf_dropliv_cross_precision = cross_val_score(clf, X_dropliv, y_dropliv, scoring = 'precision', cv = 10)
print("Random Forest Precision w/ 10-fold Cross-validation:",clf_dropliv_cross_precision.mean())

# Recall #
clf_dropliv_cross_recall = cross_val_score(clf, X_dropliv, y_dropliv, scoring = 'recall', cv = 10)
print("Random Forest Recall w/ 10-fold Cross-validation:",clf_dropliv_cross_recall.mean())

# F-1 Score #
clf_dropliv_cross_f1 = cross_val_score(clf, X_dropliv, y_dropliv, scoring = 'f1', cv = 10)
print("Random Forest F1 w/ 10-fold Cross-validation:",clf_dropliv_cross_f1.mean())

# ROC/AUC Score#
clf_dropliv_cross_rocauc = cross_val_score(clf, X_dropliv, y_dropliv, scoring = 'roc_auc', cv = 10)
print("Random Forest ROC AUC w/ 10-fold Cross-validation:",clf_dropliv_cross_rocauc)

# Confusion Matrix #
y_pred = cross_val_predict(clf, X, y, cv=10)
clf_dropliv_cross_confusion = confusion_matrix(y, y_pred)
print("Random Forest Confusion Matrix:\n", clf_dropliv_cross_confusion)

"""### ***Bagging Ensemble Classification Model***"""

# Bagging Ensemble

# Train Data
bagging_ensemble = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, random_state=0)

# Performace Metrics

# Accuracy # 
bagging_cross_accuracy = cross_val_score(bagging_ensemble, X, y, scoring = 'accuracy', cv = 10)
print("Bagging Ensemble Accuracy w/ 10-fold Cross-validation:", bagging_cross_accuracy.mean())

# Precision #
bagging_cross_precision = cross_val_score(bagging_ensemble, X, y, scoring = 'precision', cv = 10)
print("Bagging Ensemble Precision w/ 10-fold Cross-validation:",bagging_cross_precision.mean())

# Recall #
bagging_cross_recall = cross_val_score(bagging_ensemble, X, y, scoring = 'recall', cv = 10)
print("Bagging Ensemble Recall w/ 10-fold Cross-validation:",bagging_cross_recall.mean())

# F-1 Score #
bagging_cross_f1 = cross_val_score(bagging_ensemble, X, y, scoring = 'f1', cv = 10)
print("Bagging Ensemble F1 w/ 10-fold Cross-validation:",bagging_cross_f1.mean())

# ROC/AUC Score#
bagging_cross_rocauc = cross_val_score(bagging_ensemble, X, y, scoring = 'roc_auc', cv = 10)
print("Bagging Ensemble ROC AUC w/ 10-fold Cross-validation:",bagging_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(bagging_ensemble, X, y, cv=10)
bagging_cross_confusion = confusion_matrix(y, y_pred)
print("Bagging Ensemble Confusion Matrix:\n", bagging_cross_confusion)

"""### ***Boosting Ensemble Classification Model***"""

# Boosting Ensemble

# Train Data
boosting_ensemble = AdaBoostClassifier()

# Performace Metrics

# Accuracy # 
boosting_cross_accuracy = cross_val_score(boosting_ensemble, X, y, scoring = 'accuracy', cv = 10)
print("Boosting Ensemble Accuracy w/ 10-fold Cross-validation:", boosting_cross_accuracy.mean())

# Precision #
boosting_cross_precision = cross_val_score(boosting_ensemble, X, y, scoring = 'precision', cv = 10)
print("Boosting Ensemble Precision w/ 10-fold Cross-validation:",boosting_cross_precision.mean())

# Recall #
boosting_cross_recall = cross_val_score(boosting_ensemble, X, y, scoring = 'recall', cv = 10)
print("Boosting Ensemble Recall w/ 10-fold Cross-validation:",boosting_cross_recall.mean())

# F-1 Score #
boosting_cross_f1 = cross_val_score(boosting_ensemble, X, y, scoring = 'f1', cv = 10)
print("Boosting Ensemble F1 w/ 10-fold Cross-validation:",boosting_cross_f1.mean())

# ROC/AUC Score#
boosting_cross_rocauc = cross_val_score(boosting_ensemble, X, y, scoring = 'roc_auc', cv = 10)
print("Boosting Ensemble ROC AUC w/ 10-fold Cross-validation:",boosting_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(boosting_ensemble, X, y, cv=10)
boosting_cross_confusion = confusion_matrix(y, y_pred)
print("Boosting Ensemble Confusion Matrix:\n", boosting_cross_confusion)

"""### ***Gradient Tree Boosting Model***"""

# Train Data
gradient_boosting = GradientBoostingClassifier()
gradient_boosting = gradient_boosting.fit(X_train, y_train)

# Test Data
gradient_boosting_y_pred = gradient_boosting.predict(X_test)

# Performace Metrics

# Accuracy # 
boosting_cross_accuracy = cross_val_score(gradient_boosting, X, y, scoring = 'accuracy', cv = 10)
print("Gradient Boosting Ensemble Accuracy w/ 10-fold Cross-validation:", boosting_cross_accuracy.mean())

# Precision #
boosting_cross_precision = cross_val_score(gradient_boosting, X, y, scoring = 'precision', cv = 10)
print("Gradient Boosting Ensemble Precision w/ 10-fold Cross-validation:",boosting_cross_precision.mean())

# Recall #
boosting_cross_recall = cross_val_score(gradient_boosting, X, y, scoring = 'recall', cv = 10)
print("Gradient Boosting Ensemble Recall w/ 10-fold Cross-validation:",boosting_cross_recall.mean())

# F-1 Score #
boosting_cross_f1 = cross_val_score(gradient_boosting, X, y, scoring = 'f1', cv = 10)
print("Gradient Boosting Ensemble F1 w/ 10-fold Cross-validation:",boosting_cross_f1.mean())

# ROC/AUC Score#
boosting_cross_rocauc = cross_val_score(gradient_boosting, X, y, scoring = 'roc_auc', cv = 10)
print("Gradient Boosting Ensemble ROC AUC w/ 10-fold Cross-validation:",boosting_cross_rocauc.mean())

# Confusion Matrix #
y_pred = cross_val_predict(gradient_boosting, X, y, cv=10)
gradient_boosting_cross_confusion = confusion_matrix(y, y_pred)
print("Gradient Boosting Ensemble Confusion Matrix:\n", gradient_boosting_cross_confusion)

"""### ***Voting Ensemble Method***"""

# Voting Ensemble for 3 Best Classification Models (Run selected models first!)
voting_ensemble = VotingClassifier(estimators = [('Classification Tree', optimal_decision_tree), ('Boosting Ensemble', boosting_ensemble), ('Naive Bayes', naive_bayes)])

# Accuracy
for clf, label in zip([optimal_decision_tree, boosting_ensemble, naive_bayes, voting_ensemble], ['Classification Tree', 'Boosting Ensemble', 'Naive Bayes', 'Voting Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='f1', cv=10)
    print("F-1 Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

"""### ***K Nearest Neighbors***"""

knn = KNeighborsClassifier(n_neighbors=5)

# Performace Metrics

# Accuracy # 
knn_accuracy_scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print("KNN Accuracy w/ 10-fold Cross-validation:", knn_accuracy_scores.mean())

# Precision #
knn_precision_scores = cross_val_score(knn, X, y, cv=10, scoring='precision')
print("KNN Precision w/ 10-fold Cross-validation:", knn_precision_scores.mean())

# Recall #
knn_recall_scores = cross_val_score(knn, X, y, cv=10, scoring='recall')
print("KNN Recall w/ 10-fold Cross-validation:", knn_recall_scores.mean())

# F-1 Score #
knn_f1_scores = cross_val_score(knn, X, y, cv=10, scoring='f1')
print("KNN F-1 Score w/ 10-fold Cross-validation:", knn_f1_scores.mean())

# ROC/AUC Score#
knn_rocauc_scores = cross_val_score(knn, X, y, cv=10, scoring='roc_auc')
print("KNN ROC AUC w/ 10-fold Cross-validation:", knn_rocauc_scores.mean())

# Confusion Matrix #
y_pred = cross_val_predict(knn, X, y, cv=10)
knn_cross_confusion = confusion_matrix(y, y_pred)
print("KNN Confusion Matrix:\n", knn_cross_confusion)