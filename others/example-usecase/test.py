import KNearestNeighborSampling
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# used for dimension reduction for 2D visualization of multidimensional vector data
def get_embeddings_BH_tNSE(df, n_components=2, init_solution='random', random_state=0, perplexity=5):
  tsne = TSNE(n_components=n_components, method='barnes_hut', init=init_solution, random_state=random_state)
  train_sample_embedded = tsne.fit_transform(df)
  return train_sample_embedded


# Get dataset & Convert labels to integers
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(int)

# Limit dataset size to 10000
X = mnist.data.iloc[:10000, :]
y = mnist.target[:10000].tolist()

# Sample representatives to further reduce the size of dataset
train_samples = KNearestNeighborSampling.KNNSampler.sample(X, k=10, dynamic_sampling=False)
print(X.shape)
print(train_samples.shape)

try:
  train_samples['idx'] = train_samples['idx'].astype(int)
  train_samples_indices = train_samples['idx'].tolist()      # useful for plotting scatter graph
  train_samples.drop(columns=['idx'], inplace=True)
except KeyError as ke:
  print("Column is Already Dropped and Saved", ke.args[0])


# reduce dimension for plotting vectors in 2D
train_sample_embeddings = get_embeddings_BH_tNSE(train_samples, n_components=2, init_solution='random', random_state=0, perplexity=5)

# plot the 2D sampled data with 0-9 labels (labels are the dependent variable of the dataset)
labels = []
for i in train_samples_indices:
  labels.append(y[i])

label_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'gray']

# Create a scatter plot with different colors for each label
for i in range(len(labels)):
    label = labels[i]
    x = train_sample_embeddings[i, 0]
    y = train_sample_embeddings[i, 1]
    plt.scatter(x, y, c=label_colors[label])

# Add a legend to the plot
plt.legend(labels=[f'Label {i}' for i in range(10)], loc='upper left', bbox_to_anchor=(1, 1))

# Show the scatter plot
plt.show()

