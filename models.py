import numpy as np

class logreg:
    """ Logistic Regression model.

    Attributes
    ----------
    x : np.ndarray
        Array of training features, size (n, m+1) where
        the first column is 1 to account for biases
    y : np.ndarray
        Array of training targets, size (n, 1)
    x_val : np.ndarray
        Array of validation features, size (n, m+1) where
        the first column is 1 to account for biases
    y_val : np.ndarray
        Array of validation targets, size(n, 1)
    w : np.ndarray
        Array of weights, size (m+1, 1)  where
        the first column are the biases
    trained : bool
        Indicates if model has been trained
    """
    def __init__(self):
        self.trained = False

    def fit(self, x, y, x_val=None, y_val=None):
        """ Fits data to prepare for model training.

        Parameters
        ----------
        x : np.ndarray
            Array of features
        y : np.ndarray
            Array of targets
        x_val : np.ndarray, None
            Array of validation features if used
        y_val : np.ndarray, None
            Array of validation targets if used. Required if x_val
        """
        self.w = np.zeros((x.shape[1], 1))
        self.x = x
        self.y = y.reshape(y.shape[0], 1)
        if any([isinstance(x_val, np.ndarray), isinstance(y_val, np.ndarray)]) and not all([isinstance(x_val, np.ndarray), isinstance(y_val, np.ndarray)]):
            raise ValueError("Both features and targets are required if using validation parameters")
        self.x_val = x_val
        if isinstance(y_val, np.ndarray):
            self.y_val = y_val.reshape(y_val.shape[0], 1)
        else:
            self.y_val = y_val
        self.trained = False
    
    def _score(self, x):
        z = np.matmul(x, self.w)
        sigmoid = 1 / (1 + np.e**-z)
        return sigmoid
    
    def predict(self, x):
        """ Predicts likelihood of occurrence.

        Parameters
        ----------
        x : np.ndarray
            Array of features
        
        Returns
        -------
        score : np.ndarray
            Array of 0/1 predictions
        """
        score = self._score(x)
        score = np.where(score > 0.5, 1, 0)
        return score
    
    def train(self,  alpha=0.001, epochs=5000000, reg_param=0, verbose=False):
        """ Train logistic regression model

        Parameters
        ----------
        alpha : float
            Learning rate. Default 0.001
        epochs : int
            Maximum number of epochs to train for
        reg_param : float
            Regularization parameter. Default 0
        verbose : bool
            Whether or not to print loss metrics per 100 epoch
        
        Returns
        -------
        losses : list
            List of training losses over each epoch
        val_losses : list
            List of validation losses over each epoch

        Notes 
        -----
        Exits training when either validation accuracy is worse
        than previously or validation (if given, else training) loss reaches
        a stagnation point and does not change (within 6 decimal points) for 
        5 epochs
        """
        losses = []
        val_losses = []
        times_change = 0
        last_loss = 0
        last_val_loss = np.inf
        last_w = self.w.copy()
        patience = 0
        for i in range(epochs):
            score = self._score(self.x)
            score[np.where(score == 0)[0]] = 1e-6
            score[np.where(score == 1)[0]] = 1 - 1e-6
            loss = -1*np.sum(self.y*np.log(score) + (1-self.y)*np.log(1-score)) / self.y.shape[0] + \
                reg_param * np.sum(self.w[1:]**2) / (2*self.y.shape[0])
            
            if i % 100 == 0:
                if verbose:
                    print(f"********** Epoch {i} **********")
                    print_statement = f"TRAINING LOSS: {loss}"
                losses.append(loss)
            if self.x_val is not None:
                val_score = self._score(self.x_val)
                val_score[np.where(val_score == 0)[0]] = 1e-6
                val_score[np.where(val_score == 1)[0]] = 1 - 1e-6
                val_loss = -1*np.sum(self.y_val*np.log(val_score) + (1-self.y_val)*np.log(1-val_score)) / self.y_val.shape[0]
                accuracy = 1 - np.sum(abs(self.y_val - self.predict(self.x_val))) / self.y_val.shape[0]
                if i % 100 == 0:
                    if verbose:
                        print(print_statement + f", VALIDATION LOSS: {val_loss}")
                    val_losses.append(val_loss)
                if i >  10:
                    if patience != 0:
                        if (i == patience):
                            if val_loss > last_val_loss:
                                self.w = last_w.copy()
                                accuracy = 1 - np.sum(abs(self.y_val - self.predict(self.x_val))) / self.y_val.shape[0]
                                print(f"Trained in {i} Epochs. \n Validation Accuracy: {(accuracy)*100}%")
                                self.trained = True
                                return losses, val_losses
                            else:
                                patience = 0
                    else:
                        if val_loss > last_val_loss:
                            patience = i + 10
                        else:
                            last_w = self.w.copy()
                            last_val_loss = val_loss
            if (times_change == 5) and (i > 10):
                self.trained = True
                if self.x_val is not None:
                    print(f"Trained in {i} Epochs. \n Validation Accuracy: {(accuracy)*100}%")
                return losses, val_losses
               
            else:
                self.w += alpha*np.matmul(self.x.T, (self.y-score)) / self.y.shape[0]
                self.w[1:] -= alpha*reg_param * self.w[1:] / (self.y.shape[0])
                if round(loss, 6) == round(last_loss, 6):
                    times_change +=1
        if self.x_val is not None:
            accuracy = 1 - np.sum(abs(self.y_val - self.predict(self.x_val))) / self.y_val.shape[0]
            print(f"Trained in {i} Epochs. \n Validation Accuracy: {(accuracy)*100}%")
        self.trained=True
        return losses, val_losses

    def hypertune(self, detailed_output=False):
        """Pick best hyperparameters.
        
        Parameters
        ----------
        detailed_output : bool
            Returns the losses for each combination if True. Default False
        
        Notes
        -----
        Chooses learning rate and regularization parameter combination
        with the highest validation set accuracy. Hyperparameters are checked for values
        of [10, 1, 0, 0.1, 0.01, 0.001, 0.0001, 0.00001]. A learning
        rate of 0 is skipped, as this would not be a meaningful result. In the event of
        a tie, we choose the lowest validation loss. The weights for the best model will be 
        saved in the `w` attribute.
        """
        if self.x_val is None:
            raise ValueError("Cannot hypertune without a validation set, please fit one to the model")
        param_opts = [10, 1, 0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        combos = {opt: param_opts for opt in param_opts}
        loss_by_param = {}
        accuracies = {}
        best_accuracy = 0
        best_weights = {}
        for alpha, lambdas in combos.items():
            if alpha != 0:
                for lmda in lambdas:
                    print(f"Checking learning rate={alpha} and regularization parameter={lmda}")
                    losses, val_losses = self.train(alpha=alpha, reg_param=lmda)
                    accuracy = 1 - np.sum(abs(self.y_val - self.predict(self.x_val))) / self.y_val.shape[0]
                    accuracies[f"({alpha}, {lmda})"] = accuracy
                    best_weights[f"({alpha}, {lmda})"] = self.w.copy()
                    if accuracy < best_accuracy:
                        del best_weights[f"({alpha}, {lmda})"]
                    else:
                        best_accuracy = accuracy
                    loss_by_param[f"({alpha}, {lmda})"] = [losses, val_losses]
                    self.w = np.zeros((self.x.shape[1], 1))
                    self.trained=False
        optims = [key for key, value in accuracies.items() if value == max(accuracies.values())]
        if len(optims) > 1:
            lowest_losses = {}
            for opt in optims:
                lowest_losses[opt] = loss_by_param[opt][1][-1]
            optim = min(lowest_losses, key=lowest_losses.get)
        else:
            optim = optims[0]
        print(f"The optimal combination is {optim}")
        self.w = best_weights[optim]
        self.trained = True
        if detailed_output:
            return optim, max(accuracies.values()), loss_by_param
        return optim, max(accuracies.values())
