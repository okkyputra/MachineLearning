# Importing the necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
# Load iris dataset
iris = load_iris()


# Identifying the features (independent variables) and the dependent variable
print("Features: ", iris.feature_names)
print("Dependent Variable: Species of Iris plant")

# Create dataframe
df = pd.DataFrame(data=iris.data, columns = iris.feature_names)

# Create matrix of features (X) and dependent variable vector (y)
X = df.values
y = iris.target

# Print your feature matrix (X) and dependent variable vector (y)
print(X)
print(y)