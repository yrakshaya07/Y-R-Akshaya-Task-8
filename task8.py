import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

df = pd.read_csv(r"C:\Users\AKSHAYA\Downloads\Mall_Customers.csv")  
print("Original Data Shape:", df.shape)
print(df.head())

df_numeric = df.select_dtypes(include=[np.number])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, color='purple', marker='o', linestyle='-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method For Optimal K', fontsize=14)
plt.grid(True)
plt.show()

optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(scaled_data)

sil_score = silhouette_score(scaled_data, cluster_labels)
print(f"Silhouette Score for K={optimal_k}: {sil_score:.3f}")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

custom_cmap = ListedColormap(['purple', 'skyblue', 'orange'])

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap=custom_cmap)
plt.title(f'K-Means Clustering (K={optimal_k}) - PCA 2D View', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()