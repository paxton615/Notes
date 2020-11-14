import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class ada_boost_class:
    
    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3):
        """
        arguments: base_estimator, number of estimators
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
    
    def _calculate_error_rate(self, w, miss):
        """
        arguments: weights and mask for misclassified data points (miss)
        returns: error rate
        """
        error_rate = np.dot(w, miss) / w.sum()
        return error_rate
    
    def _calculate_alpha(self, error_rate):
        """
        calculates the alpha value at current iteration step
        argument: error rate
        returns: alpha_m
        """
        alpha_t = np.log((1. - error_rate) / error_rate)
        return alpha_t
    
    def _indicator_function(self, y, pred):
        """
        arguments: target variable and predicted values
        returns: 1 if disagreement, 0 else
        """
        return (pred != y)*1
    
    def _binary_converter(self, y):
        """
        Convert binary class labels to 1/-1 (one of the labels must be codes as 1)
        """
        return np.array([x if x == 1 else -1 for x in y])
    
    def _update_importance_weights(self, w, miss, alpha_t):
        """
        arguments: current weights, mask for misclassified observations, alpha_m
        returns: w, the updated weight
        """
        w = np.multiply(w, np.exp([float(x) * alpha_t for x in miss]))
        return w

    def _update_predictions(self, pred, pred_t, alpha_t):
        """
        update the predictions with the ones of the current model
        arguments: previous aggregated predictions (pred), 
                   predictions at current iteration,
                   alpha_m
        returns: updated predictions
        """

        pred = [sum(p) for p in zip(pred, [m * alpha_t for m in pred_t])]
        return pred
    
    def _get_score(self, y, pred, score=accuracy_score):
        """
        arguments: target values and predicted values
        returns: score
        """
        return score(y, pred)
    
    def _model_predictions(self, X_train, y_train, X_test, y_test, sample_weight=None):
        """
        arguments: model, training and test sets, sample weights
        body: fit the model and make predicitions
        returns: training predictions and test predictions
        """
        self.base_estimator.fit(X_train, y_train, sample_weight=sample_weight)
        pred_train = self.base_estimator.predict(X_train)
        pred_test = self.base_estimator.predict(X_test)
        return pred_train, pred_test
    
    def _generic_model(self, X_train, y_train, X_test, y_test):
        """
        arguments: train and test sets for X, y and a model
        fits the model on the training data and obtains train/test predictions
        returns: train/test scores
        """
        pred_train, pred_test = self._model_predictions(
                X_train, y_train, X_test, y_test)
        
        return self._get_score(y_train, pred_train), \
            self._get_score(y_test, pred_test)
    
    
    
    def predict(self, X_train, y_train, X_test, y_test):
        """
        arguments: training/test set predictors and labels
        returns: predicted labels after iteration over n_estimators
        """
        
        # measure size of train/test sets
        n_train, n_test = len(X_train), len(X_test)
        # Initialize weights
        w = np.ones(n_train) / n_train
        # Initialize train/test predictions
        pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
        # convert labels to 1/-1
        ytilde_train = self._binary_converter(y_train)
        ytilde_test = self._binary_converter(y_test)
        
        for i in range(self.n_estimators):
            # Fit a classifier with the current weights
            pred_train_i, pred_test_i = self._model_predictions(
                X_train, y_train, X_test, y_test, sample_weight=w)
            # convert predicted labels to 1/-1
            pred_train_i = self._binary_converter(pred_train_i)
            pred_test_i = self._binary_converter(pred_test_i)
            # Indicator function
            miss = self._indicator_function(pred_train_i, ytilde_train)
            # Error
            err_m = self._calculate_error_rate(w, miss)
            # Alpha
            alpha_m = self._calculate_alpha(err_m)
            # update weights
            w = self._update_importance_weights(w, miss, alpha_m)
            # Add to prediction
            pred_train = self._update_predictions(pred_train, pred_train_i, alpha_m)
            pred_test = self._update_predictions(pred_test, pred_test_i, alpha_m)

        # get the sign of train/test predictions
        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        
        return pred_train, pred_test
        
    def score(self, X_train, y_train, X_test, y_test):
    
        ytilde_train = self._binary_converter(y_train)
        ytilde_test = self._binary_converter(y_test)
        pred_train, pred_test = self.predict(
            X_train, y_train, X_test, y_test)
        
        # Return train/test scores
        return self._get_score(ytilde_train, pred_train), \
            self._get_score(ytilde_test, pred_test)