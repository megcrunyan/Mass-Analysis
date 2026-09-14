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

$$\sigma(z) = \frac{1}{(1+e^{-z})}$$

$$J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} y^{(i)}\log(\sigma(z)) + (1-y^{(i)})\log(1-\sigma(z))$$

$$\nabla_\theta \sigma(z) = \sigma(z)(1 - \sigma(z))\nabla_\theta z$$

$$\nabla_\theta z = x^{(i)}$$

 Then the gradient of the loss function $J$ becomes: 
$$\nabla_\theta J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} (y^{(i)} \frac{\sigma(z)(1-\sigma(z))}{\sigma(z)} + (1 - y^{(i)})\frac{-\sigma(z)(1-\sigma(z))}{(1-\sigma(z))})x^{(i)}$$
Simplifying, we get:

$$\nabla_\theta J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \sigma(z))x^{(i)} = -\frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \frac{1}{(1+e^{-z})})x^{(i)}$$

The weight update equation is given as: $$\theta = \theta - \alpha \nabla_\theta J(\theta)$$
Plugging in our values, we get:
$$\theta = \theta + \frac{\alpha}{n} \sum_{i=1}^{n}(y^{(i)}-\frac{1}{(1+e^{-z})})x^{(i)}$$

With L2 regularization, we have the loss function:

$$J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} (y^{(i)}\log(\sigma(z)) + (1-y^{(i)})\log(1-\sigma(z))) + \frac{\lambda}{n}||\theta||^2_{2}$$


$$\nabla_\theta J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \frac{1}{(1+e^{-z})})x^{(i)} + \frac{\lambda}{n}2\theta$$

and the update equation:

$$\theta = \theta + \frac{\alpha}{n}\sum_{i=1}^{n}(y^{(i)}-\frac{1}{(1+e^{-z})})x^{(i)} - \frac{\lambda \alpha}{n} 2 \theta$$

#### From Scratch
Logistic Regression class has been implemented with options for regularization and a hyperparameter tuning option in which the optimal hyperparameters are selected.

#### `Scikit-learn`
The logistic regression solver and stochastic gradient descent methods have both been implemented, with SGD being optimized by a Grid Search. 

### Neural Network
Many of the principles from our from scratch logistic regression implementation, but now with more layers.
To demonstrate the derivation, let's look at a simple two-layer neural network using the same binary cross-entropy loss function as above. 

$$z_1^{(i)} = \theta^T_1x^{(i)} + b_1$$

$$a_1^{(i)} = \sigma(z_1^{(i)}) = \frac{1}{(1+e^{-z_1^{(i)}})}$$

$$z_2^{(i)} = \theta^T_2a_1^{(i)} + b_2$$

$$a_2^{(i)} = \sigma(z_2^{(i)}) = \frac{1}{(1+e^{-z_2^{(i)}})}$$

$$J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} y^{(i)}\log(a_2^{(i)}) + (1-y^{(i)})\log(1-a_2^{(i)})$$

In general, we know that, for the $k^{th}$ layer

$$\frac{da_k}{dz_k} = \sigma(z_k)(1 - \sigma(z_k))$$

So then, working our way out through the update equations, we get

$$\nabla_{\theta_2} J(\theta) = \frac{dJ(\theta)}{da_2}\frac{da_2}{dz_2}\frac{dz_2}{d\theta_2} = \frac{-1}{n} \sum_{i=1}^{n} (y^{(i)} - a_2^{(i)})a_1^{(i)} $$

$$\nabla_{b_2} J(\theta) = \frac{dJ(\theta)}{da_2}\frac{da_2}{dz_2}\frac{dz_2}{db_2} = \frac{-1}{n} \sum_{i=1}^{n} (y^{(i)} - a_2^{(i)}) $$
$$\nabla_{\theta_1} J(\theta) = \frac{dJ(\theta)}{da_2}\frac{da_2}{dz_2}\frac{dz_2}{da_1}\frac{da_1}{dz_1}\frac{dz_1}{d\theta_1} = \frac{-1}{n} \sum_{i=1}^{n} (y^{(i)} - a_2^{(i)})\theta_2\sigma(z_1)(1 - \sigma(z_1))x^{(i)} $$

$$\nabla_{b_1} J(\theta) = \frac{dJ(\theta)}{da_2}\frac{da_2}{dz_2}\frac{dz_2}{da_1}\frac{da_1}{dz_1}\frac{dz_1}{db_1} = \frac{-1}{n} \sum_{i=1}^{n} (y^{(i)} - a_2^{(i)})\theta_2\sigma(z_1)(1 - \sigma(z_1)) $$

Then the weight updates for layer k become

$$\theta_k = \theta_k - \alpha \nabla_{\theta_k} J(\theta)$$

and the biases become

$$b_k = b_k - \alpha \nabla_{b_k} J(\theta)$$

#### From Scratch
A neural network class has been implemented with options to select the number of layers and neurons at each layer with a hyperparameter tuning option in which the optimal learning rate is selected.

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
Many of the concepts and content in this project were learned from CS229: Machine Learning at Stanford University.

Specifically for early stopping conditions, the method implemented was selected from a few forum posts below:

The Lazy Log. (2016 August 22). *How to use Early Stopping Properly for Training Deep Neural Network?* Cross Validated Stack Exchange. https://stats.stackexchange.com/questions/231061/how-to-use-early-stopping-properly-for-training-deep-neural-network

qmeeus. (2018 August 20). *Early Stopping on Validation Loss or on Accuracy?* Data Science Stack Exchange. https://datascience.stackexchange.com/questions/37186/early-stopping-on-validation-loss-or-on-accuracy

Specifically for the Xavier weight initialization in the neural network, the method implemented was selected from a few blog posts below: 

GeeksforGeeks. (2025 July 23). *Weight Initialization Techniques for Deep Neural Networks*. GeeksforGeeks.
https://geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/

GeeksforGeeks. (2025 July 23). *Xavier initialization*. GeeksforGeeks.
https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
