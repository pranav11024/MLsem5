import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# A1: Linear Regression with one attribute
def linear_regression_one_attr(X_train, y_train):
    reg = LinearRegression().fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    return y_train_pred, reg

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def linear_regression_multi_attr(X_train, y_train):
    reg = LinearRegression().fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    return y_train_pred, reg

def kmeans_clustering(X_train, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_train)
    return kmeans.labels_, kmeans.cluster_centers_

def clustering_scores(X_train, labels):
    silhouette = silhouette_score(X_train, labels)
    ch_score = calinski_harabasz_score(X_train, labels)
    db_score = davies_bouldin_score(X_train, labels)
    return silhouette, ch_score, db_score

def kmeans_clustering_different_k(X_train, k_values):
    scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)
        silhouette = silhouette_score(X_train, kmeans.labels_)
        ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
        db_score = davies_bouldin_score(X_train, kmeans.labels_)
        scores.append((k, silhouette, ch_score, db_score))
    return scores

def elbow_plot(X_train, k_values):
    distortions = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k).fit(X_train)
        distortions.append(kmeans.inertia_)
    plt.plot(k_values, distortions)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

#Main
data = pd.read_excel(r"CE_vector_v3.xlsx")
X_train = data[["Control_Flow_Structures_Vector_0", "Control_Flow_Structures_Vector_1"]]
y_train = data["Final_Marks"]
# A1
y_train_pred_one, reg_one = linear_regression_one_attr(X_train[["Control_Flow_Structures_Vector_0"]], y_train)
print("A1 - Linear Regression (one attribute) Predictions:", y_train_pred_one)

# A2
mse, rmse, mape, r2 = calculate_metrics(y_train, y_train_pred_one)
print(f"A2 - Metrics: MSE={mse}, RMSE={rmse}, MAPE={mape}, R2={r2}")

# A3
y_train_pred_multi, reg_multi = linear_regression_multi_attr(X_train, y_train)
print("A3 - Linear Regression (multiple attributes) Predictions:", y_train_pred_multi)

# A4
labels, centers = kmeans_clustering(X_train)
print("A4 - K-means Clustering Labels:", labels)
print("A4 - K-means Clustering Centers:", centers)

# A5
silhouette, ch_score, db_score = clustering_scores(X_train, labels)
print(f"A5 - Clustering Scores: Silhouette={silhouette}, CH Score={ch_score}, DB Score={db_score}")

# A6
k_values = range(2, 10)
clustering_results = kmeans_clustering_different_k(X_train, k_values)
print("A6 - K-means Clustering Results for different k values:", clustering_results)

# A7
elbow_plot(X_train, k_values)
