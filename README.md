## Dimensionality-Reduction

### Abstract

It’s difficult to visualize data and understand the relationships of it in high dimensional space. As dimensions get higher, the curse of dimensionality occurs. Hughes phenomenon which is when the dimensions increase, the number of samples needed for good prediction also increase logarithmically. Therefore, we use techniques like Principal Component Analysis or t-SNE to reduce these dimensions and visualize them in lower dimensions. Visualization isn’t the only benefit. High dimensional data can cause overfitting because models might capture noise rather than meaningful patterns. Dimensionality reduction simplifies the feature set, helping to focus on the most informative parts of the data. Fewer dimensions speeds up training and prediction, especially for complex models like neural networks. It mitigates the curse of dimensionality: data points become sparse, making it difficult to find meaningful relationships in the data. Dimensionality Reduction has applications from data visualization, preprocessing for machine learning, feature extraction, noise reduction to compression. Usually, a fraction of the feature set contains the majority of explained variance, meaning you can use less features to convey the important information.

### Introduction

My goal is to apply dimensionality reduction to the MNIST dataset, a collection of greyscale handwritten digits, with two main objectives:  

1. Data Visualization: Reduce the dataset to 2 dimensions to visualize the relationships between data points and identify the features that contribute most to the variance.  
2. Data Preprocessing: Simplify the feature set by reducing dimensionality, making the data more efficient and effective for training a neural network.  

The MNIST dataset comes with a training set of 60,000 images and a testing set of 10,000 images. Each image is a 28 x 28 pixel square, so each image contains 784 (28x28) features that contain the images’ pixel values along with a label (a number between 0 and 9).

I used two dimensionality reduction techniques, Principal Component Analysis and t-distributed Stochastic Neighbor Embedding (t-SNE). 

### Proposed Methodology

I’m using both PCA and t-SNE to get a further understanding of their algorithm. 

Principal Component Analysis (PCA) is an unsupervised technique used for dimensionality reduction. It identifies directions, called principal components, that capture the most variance in the data. PCA works best when features are highly correlated because it transforms this redundancy into a smaller set of uncorrelated components, retaining as much of the original information as possible. The method is based on a linear transformation that reduces the dimensionality of the data. It begins by finding the first principal component, which explains the maximum variance in the dataset. Subsequent components capture the remaining variance, with each being uncorrelated and orthogonal (perpendicular) to the others. This is achieved through a linear orthogonal transformation, which rotates the original feature axes to align with the directions of maximum spread in the data. The resulting principal components are linear combinations of the original features, making them uncorrelated and ordered by the amount of variance they explain.

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction technique designed to preserve local structure in data, making it ideal for visualizing complex high-dimensional datasets. Unlike PCA, which focuses on preserving global structure, t-SNE excels at revealing the "neighborhoods" of data points.  

t-SNE maps high-dimensional data to a lower-dimensional space (typically 2D or 3D) by prioritizing the preservation of local relationships. In high dimensions, distances can lose meaningful context, so t-SNE models the similarity of points using a Gaussian probability distribution around each data point. Nearby points with higher similarity scores are considered neighbors.  

To project the data into lower dimensions, t-SNE initializes points randomly in the target space and models their similarities using a t-distribution, which has heavier tails than a Gaussian. This choice helps prevent crowding of points in the lower-dimensional space. The algorithm then uses gradient descent to minimize the difference (or KL divergence) between the high-dimensional and low-dimensional similarity distributions, ensuring that the local relationships are faithfully represented.  

### Analysis and Results

The dataset was provided in an IDX format, a binary structure for high-dimensional arrays, which I converted into NumPy arrays and subsequently CSV files for processing. To ensure consistent scaling, I standardized the feature set. This standardization balanced contributions from all features, improved convergence during iterative optimization (e.g., for t-SNE), and ensured variance calculations were more representative.

Using Principal Component Analysis (PCA), I calculated the covariance matrix and from that the eigenvalues and eigenvectors to understand the directions and magnitudes of linear transformations in the data. It was determined that 300 principal components are required to retain 90% of the dataset's variance, highlighting its complexity. Reducing the data to two dimensions captures only a small fraction of the variance and cannot fully represent the original dataset.

To visualize the eigenvectors, I reshaped the first two principal components into the original image size. The first two components only represent a small portion of the variance, so it reaffirmed little to no image information can be gained by only 2 components’ directions. 

Using the first 300 eigenvectors, I reconstructed the MNIST images to assess how much of the original data was retained. The reconstructed images were fairly decipherable, confirming that these components preserved significant structural information despite the dimensionality reduction.

To explore potential patterns, I plotted the first two principal components in a scatter plot, labeling the data by digit. While the scatter plot showed limited separation due to the small variance explained by these components, distinct clusters emerged for most digits. Some random variance existed, most likely because the variance captured by these principal components does not represent all of the dataset’s variance. 

I then applied t-SNE to compare its results with PCA. t-SNE revealed better-separated clusters, but with scattered points potentially indicating digit similarity, noise, or projection loss . Unlike PCA, t-SNE plots do not preserve global distances and rely heavily on hyperparameters like perplexity, which balances attention between local and global relationships. The results showed that higher perplexity values emphasized global patterns, while lower values revealed finer local structure. However, the distances and sizes of clusters in t-SNE are not meaningful due to how it adjusts for density variations, making direct interpretation challenging. 

### Conclusions 

The analysis highlights the trade-offs in dimensionality reduction techniques. PCA, while linear and interpretable, struggles to represent the complexity of high-dimensional data like MNIST in only two dimensions. Nevertheless, it retains enough structure to provide a coarse visualization of global patterns. In contrast, t-SNE excels at uncovering local structures and separating clusters, but its dependence on hyperparameters and lack of interpretable distances limits its utility for broader insights.

The findings suggest that while PCA is useful for understanding global variance and reconstructing images, t-SNE provides a complementary perspective for visualizing local structure. 

