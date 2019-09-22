from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os


class TemplateClassifier(BaseEstimator, ClassifierMixin):

     def __init__(self, random_state=seed):
         self.model = LogisticRegression(random_state=random_state)

     def fit(self, X, y):

         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         # Store the classes seen during fit
         self.classes_ = unique_labels(y)

         self.X_ = X
         self.y_ = y
         
         self.model.fit(self.X_, self.y_)

         return self

     def predict(self, X):

         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_'])

         # Input validation
         X = check_array(X)

         self.prediction = self.model.predict(X)
         return self.prediction