from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
import mord

from dl_models import LinearBase, MLPBase
from pom_scratch import POMHead
from adjacent_model import AdjacentHead
from coral import OrdinalHead
from classification_head import ClassificationHead
from combined_model import CombinedModel

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
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")