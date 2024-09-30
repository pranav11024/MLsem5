import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# A2: Hyperparameter tuning for Perceptron
def tune_perceptron(X_train, y_train, X_test, y_test):
    param_grid = {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [1000, 5000, 10000]
    }
    perceptron = Perceptron()
    search = RandomizedSearchCV(perceptron, param_grid, n_iter=10, cv=3)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    return best_model, accuracy_score(y_test, predictions), precision_score(y_test, predictions, average='weighted'), recall_score(y_test, predictions, average='weighted'), f1_score(y_test, predictions, average='weighted')

# A2: Hyperparameter tuning for MLP Network
def tune_mlp_network(X_train, y_train, X_test, y_test):
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01]
    }
    mlp = MLPClassifier(max_iter=1000)
    search = RandomizedSearchCV(mlp, param_grid, n_iter=10, cv=3)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    return best_model, accuracy_score(y_test, predictions), precision_score(y_test, predictions, average='weighted'), recall_score(y_test, predictions, average='weighted'), f1_score(y_test, predictions, average='weighted')

# A3: Tabulation with various classifiers
def compare_classifiers(X_train, y_train, X_test, y_test):
    classifiers = {
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'NaiveBayes': GaussianNB()
    }
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision': precision_score(y_test, predictions, average='weighted'),
            'Recall': recall_score(y_test, predictions, average='weighted'),
            'F1 Score': f1_score(y_test, predictions, average='weighted')
        }
    return results


file_path = 'Scaled_CE_vector_v3.xlsx'
df = pd.read_excel(file_path)
df2=pd.read_excel("CE_vector_v3.xlsx")
# Separate the features (2304-dimensional summed vectors)
X = df[[f'fused_vector_v{i}' for i in range(2304)]]

# Extract the 'Final_Marks' column
y = df2['Final_Marks']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Call A2: Perceptron with hyperparameter tuning
perceptron_model, acc_perc, prec_perc, rec_perc, f1_perc = tune_perceptron(X_train, y_train, X_test, y_test)
print(f"Perceptron Accuracy: {acc_perc}, Precision: {prec_perc}, Recall: {rec_perc}, F1 Score: {f1_perc}")
# Call A2: MLP with hyperparameter tuning
mlp_model, acc_mlp, prec_mlp, rec_mlp, f1_mlp = tune_mlp_network(X_train, y_train, X_test, y_test)
print(f"MLP Accuracy: {acc_mlp}, Precision: {prec_mlp}, Recall: {rec_mlp}, F1 Score: {f1_mlp}")
# Call A3: Comparison with other classifiers
results = compare_classifiers(X_train, y_train, X_test, y_test)
print("Classifier Results:")
for clf, metrics in results.items():
    print(f"{clf} -> Accuracy: {metrics['Accuracy']}, Precision: {metrics['Precision']}, Recall: {metrics['Recall']}, F1 Score: {metrics['F1 Score']}")
