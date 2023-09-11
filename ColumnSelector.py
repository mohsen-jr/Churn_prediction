from sklearn.base import BaseEstimator, TransformerMixin


## create a custom column selector
class ColumnSelector(BaseEstimator, TransformerMixin):
    """select specific columns of a given dataset"""
    def __init__(self, subset):
        self.subset = subset
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.loc[:, self.subset]
    
    def fit_transform(self, X, y=None):
        return X[self.subset]