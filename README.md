# Mass Analysis

Investigating different machine learning methods to classify breast cancer from tumors.

## Datasets Used
A CSV of 30 features of tumors with a column of malignant/benign diagnoses, cited below:
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

## Methods Investigated
### Logistic Regression
In our from scratch implementation, we will be using Gradient Descent to help find a set of weights to estimate the likelihood of malignant tumors in this data, using binary cross-entropy as a loss function.
To help with the derivation, we define a few functions here:

$$z = \theta^Tx^{(i)} + \theta_0$$

$$\sigma(z) = 1/(1+e^{-z})$$

$$J(\theta) = -1/n \sum_{i=1}^{n} y^{(i)}log(\sigma(z)) + (1-y^{(i)})log(1-\sigma(z))$$

$$\nabla_\theta \sigma(z) = \sigma(z)(1 - \sigma(z))\nabla_\theta z$$

$$\nabla_\theta z = x^{(i)}$$

 Then the gradient of the loss function $J$ becomes: 
 
$$\nabla_\theta J(\theta) = -1/n \sum_{i=1}^{n} (y^{(i)} 1/\sigma(z)\sigma(z)(1-\sigma(z))x^{(i)} + (1 - y^{(i)})1/(1-\sigma(z))-\sigma(z)(1-\sigma(z))x^{(i)})$$

Simplifying, we get:

$$\nabla_\theta J(\theta) = -1/n \sum_{i=1}^{n} (y^{(i)} - \sigma(z))x^{(i)} = -1/n \sum_{i=1}^{n} (y^{(i)} - 1/(1+e^{-z}))x^{(i)}$$

The weight update equation is given as: $$w = w - \alpha \nabla_\theta J(\theta)$$
Plugging in our values, we get:
$$w = w + \alpha / n \sum_{i=1}^{n}(y^{(i)}-1/(1+e^{-z}))x^{(i)}$$

With L2 regularization, we have the loss function:

$$J(\theta) = -1/n \sum_{i=1}^{n} (y^{(i)}log(\sigma(z)) + (1-y^{(i)})log(1-\sigma(z))) + (\lambda/n)||w||_2^2$$

$$\nabla_\theta J(\theta) = -1/n \sum_{i=1}^{n} (y^{(i)} - 1/(1+e^{-z}))x^{(i)} + (\lambda/n)2w^{(i)}$$

and the update equation

$$w = w + \alpha(1/n)\sum_{i=1}^{n}(y^{(i)}-1/(1+e^{-z}))x^{(i)} + (\lambda/n) 2 w$$

#### From Scratch
Logistic Regression class has been implemented with options for regularization and a hyperparameter tuning option in which the optimal hyperparameters are selected.

#### `Scikit-learn`
The logistic regression solver and stochastic gradient descent methods have both been implemented, with SGD being optimized by a Grid Search.

### Neural Network
#### From Scratch
TO BE COMPLETED

#### `pytorch`
TO BE COMPLETED


## Road Map
- Logistic Regression
    - Compare outputs of Logistic Regression to `scikit-learn`'s implementation. We will use accuracy, precision, recall, and f1 score as model metrics.
- Neural Networks
    - Implement a neural network from scratch and one using pytorch. Compare outputs of the neural networks to  each other using accuracy, precision, recall, and f1 score.
- Model Selection
    - Evaluate all models, compare metrics and differences, and determine which would be recommended for identifying breast cancer in this dataset.

## References
Many of the concepts and content in this project was learned from CS229: Machine Learning at Stanford University.

Specifically for early stopping conditions, the method implemented was selected from a few forum posts below:
The Lazy Log. (2016 August 22). *How to use Early Stopping Properly for Training Deep Neural Network?* Cross Validated Stack Exchange. https://stats.stackexchange.com/questions/231061/how-to-use-early-stopping-properly-for-training-deep-neural-network

qmeeus. (2018 August 20). *Early Stopping on Validation Loss or on Accuracy?* Data Science Stack Exchange. https://datascience.stackexchange.com/questions/37186/early-stopping-on-validation-loss-or-on-accuracy

