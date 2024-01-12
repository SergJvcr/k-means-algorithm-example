# Import standard operational packages
import numpy as np
import pandas as pd
# Important tools for modeling and evaluation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Import visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

penguins = pd.read_csv('google_data_analitics\\penguins.csv')
print(penguins.head(10))

# Data exploration
# Explore data
print(penguins['species'].unique())
print(penguins['species'].value_counts(dropna=False))
# Check for missing values
print(penguins.isna().sum())

penguins_subset = penguins.dropna(axis=0).reset_index(drop=True) # drop rows with missing values

print(penguins_subset.isna().sum()) # check for missing values after drop it

# Encode data
penguins_subset['sex'] = penguins_subset['sex'].str.upper()
print(penguins_subset.head(5))

# Convert `sex` column from categorical to numeric
penguins_subset = pd.get_dummies(penguins_subset, drop_first=True, columns=['sex'])
print(penguins_subset.head(5))

# Drop the column `island`
penguins_subset = penguins_subset.drop(['island'], axis=1)
print(penguins_subset.head(2))

# Scale the features
# Because K-means uses distance between observations as its measure of similarity, 
# it's important to scale the data before modeling
X = penguins_subset.drop(['species'], axis=1) # exclude `species` variable from X
print(X.head(2))

#Assign the scaled data to variable `X_scaled`
X_scaled = StandardScaler().fit_transform(X)
print(X_scaled[:2, :]) # to show the first two rows of scaled dataset

# Data modeling
# Fit K-means and evaluate inertia for different values of K
num_clusters = [i for i in range(2, 11)]

def kmeans_inertia(num_clusters, x_values):
    """
    Fits a KMeans model for different values of k.
    Calculates an inertia score for each k value.

    Args:
        num_clusters: (list of ints)  - The different k values to try
        x_vals:       (array)         - The training data

    Returns: 
        inertia:      (list)          - A list of inertia scores, 
                                        one for each value of k
    """
    inertia = []
    
    for num in num_clusters:
        kmeans = KMeans(n_clusters=num, random_state=42, init='k-means++', n_init=10)
        kmeans.fit(x_values)
        inertia.append(kmeans.inertia_)
        
    return inertia

# Return a list of inertia for k = 2 to 10
inertia = kmeans_inertia(num_clusters, X_scaled)
print(f'The inertia values for K from 2 to 10: {inertia}')

# Create a line plot
plot = sns.lineplot(x=num_clusters, y=inertia, marker='o', color='green')
plot.set_title('The line plot of num_clusters vs inertia')
plot.set_xlabel('Number of clusters')
plot.set_ylabel('Inertia')
plot.axvline(x=6, color='red')
plt.show()

# Results and evaluation
# Evaluate silhouette score for comparing with inertia
# Write a function to return a list of each k-value's score
def kmeans_sil_score(num_clusters, x_values):
    """
    Fits a KMeans model for different values of k.
    Calculates a silhouette score for each k value

    Args:
        num_clusters: (list of ints)  - The different k values to try
        x_vals:       (array)         - The training data

    Returns: 
        sil_score:    (list)          - A list of silhouette scores, 
                                        one for each value of k
    """
    sil_score = []
    
    for num in num_clusters:
        kmeans = KMeans(n_clusters=num, random_state=42, init='k-means++', n_init=10)
        kmeans.fit(x_values)
        sil_score.append(silhouette_score(x_values, kmeans.labels_))
    
    return sil_score


kmeans_silhouette_score = kmeans_sil_score(num_clusters, X_scaled)
print(f'The silhouette score values for K from 2 to 10: {kmeans_silhouette_score}')

# Create a line plot
plot2 = sns.lineplot(x=num_clusters, y=kmeans_silhouette_score, marker='o', color='orange')
plot2.set_title('The line plot of num_clusters vs sil_score')
plot2.set_xlabel('Number of clusters')
plot2.set_ylabel('Silhouette score')
plot2.axvline(x=6, color='red')
plt.show()

# Optimal k-value = 6 (Inertia and Silhouette score show)
# Fit a 6-cluster model
kmeans_6_clusters = KMeans(n_clusters=6, random_state=42, init='k-means++')
kmeans_6_clusters.fit(X_scaled)

# Print unique labels
print(kmeans_6_clusters.labels_)
print(f'Unique labels: {np.unique(kmeans_6_clusters.labels_)}')

# Create a new column `cluster` in the first dataset
penguins_subset['cluster'] = kmeans_6_clusters.labels_
print(penguins_subset.head(10))

# Verify if any `cluster` can be differentiated by `species`
print(penguins_subset.groupby(by=['cluster', 'species']).size())

penguins_subset.groupby(by=['cluster', 'species']).size().plot.bar(title='Clusters differentiated by species',
                                                                   figsize=(6, 5),
                                                                   ylabel='Size',
                                                                   xlabel='(Cluster, Species)')
plt.show()

# Verify if each `cluster` can be differentiated by `species' AND `sex_MALE`
penguins_subset.groupby(by=['cluster', 'species', 'sex_MALE']).size().sort_values(ascending=False)

penguins_subset.groupby(by=['cluster','species','sex_MALE']).size().unstack(level = 'species', fill_value=0).plot.bar(title='Clusters differentiated by species and sex',
                                                                                                                      figsize=(6, 5),
                                                                                                                      ylabel='Size',
                                                                                                                      xlabel='(Cluster, Sex)')
plt.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()
