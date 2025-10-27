from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import mord

from .base_models import LinearBase, MLPBase
from .pom_head import POMHead
from .adjacent_head import AdjacentHead
from .coral_head import OrdinalHead
from .classification_head import ClassificationHead
from .combined_model import CombinedModel

class OrdinalSVC(BaseEstimator, ClassifierMixin):
    """
    Support Vector Classifier adapted for Ordinal Regression using the cumulative link approach.
    Trains k-1 binary classifiers to predict P(y > c) for each category c.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.clfs = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError("Cannot perform ordinal classification with less than 2 classes.")

        # Train n_classes_ - 1 binary classifiers
        for i in range(self.n_classes_ - 1):
            # Create binary target: 1 if y > class_i, 0 otherwise
            binary_y = (y > self.classes_[i]).astype(int)
            
            clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, random_state=self.random_state, probability=True)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict(self, X):
        if not self.clfs:
            raise RuntimeError("Model not fitted yet.")

        # Get probabilities P(y > c) for each classifier
        # Each clf.predict_proba returns [P(y <= c), P(y > c)]
        # We want P(y > c), which is the second column
        cumulative_probs = np.array([clf.predict_proba(X)[:, 1] for clf in self.clfs]).T

        # Convert cumulative probabilities to class predictions
        # The predicted class is the count of how many P(y > c) are > 0.5
        # This effectively finds the first 'c' for which P(y > c) is false (i.e., P(y <= c) is true) 
        predictions = np.sum(cumulative_probs > 0.5, axis=1)
        return self.classes_[predictions]

    def _more_tags(self):
        return {'binary_only': False, 'requires_y': True, 'poor_score': True, 'multioutput': False}

def get_dl_model(base_model_name, head_model_name, is_regression, **kwargs):
    """Assembles and returns a combined PyTorch model."""
    input_size = kwargs['input_size']
    num_classes = kwargs['num_classes']

    # 1. Create the base model
    if base_model_name.lower() == 'linear':
        base_model = LinearBase(input_size)
    elif base_model_name.lower() == 'mlp':
        base_model = MLPBase(input_size)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")

    # 2. Create the head model
    head_input_size = base_model.output_size
    if head_model_name.lower() == 'pom':
        head_model = POMHead(head_input_size, num_classes)
    elif head_model_name.lower() == 'adjacent':
        head_model = AdjacentHead(head_input_size, num_classes)
    elif head_model_name.lower() in ['coral', 'corn']:
        head_model = OrdinalHead(head_input_size, num_classes)
    elif head_model_name.lower() in ['mlp', 'mlp-emd']:
        head_model = ClassificationHead(head_input_size, num_classes)
    else:
        raise ValueError(f"Unknown head model: {head_model_name}")

    # 3. Combine them
    return CombinedModel(base_model, head_model)

def get_sklearn_model(model_name, is_regression):
    """Returns a standard scikit-learn model."""
    if model_name.lower() == 'decisiontree':
        return DecisionTreeRegressor(random_state=42) if is_regression else DecisionTreeClassifier(criterion='entropy', random_state=42)
    elif model_name.lower() == 'svm':
        return SVR() if is_regression else SVC(random_state=42)
    elif model_name.lower() == 'clm':
        if is_regression: raise ValueError("CLM is not for regression.")
        return mord.OrdinalRidge(alpha=0)
    elif model_name.lower() == 'svor':
        if is_regression: raise ValueError("SVOR is not for regression.")
        return OrdinalSVC(random_state=42)
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")
