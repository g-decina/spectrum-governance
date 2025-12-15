import numpy as np
import sys

try:
    import torch
except ImportError:
    pass

from sklearn.base import BaseEstimator

class DLAdapter(BaseEstimator):
    def __init__(self, dl_model):
        self.dl_model = dl_model
        
        # Determine framework and device
        if "torch" in self.modules:
            self.is_torch = isinstance(dl_model, torch.nn.Module)
        else:
            self.is_torch = False
        
        is_sklearn = isinstance(dl_model, BaseEstimator)
        
        self.is_tf = (not self.is_torch) and (not is_sklearn) and hasattr(dl_model, 'predict')
        
        if self.is_torch:
            self._device = next(dl_model.parameters()).device
        elif self.is_tf:
            self._device = "CPU" # Assume TF handles its own GPU allocation
            
        self._estimator_type = dl_model._estimator_type if hasattr(dl_model, '_estimator_type') else 'classifier' 
        
        self.fitted_ = True # So check_is_fitted returns True
    
    def fit(self, X, y=None): 
        # CRITICAL: NO-OP. The DL model is assumed to be pre-trained.
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        
        if self.is_torch:
            self.dl_model.eval()
            
            X_tensor = torch.from_numpy(X).float().to(self._device)
            
            with torch.no_grad():
                output_tensor = self.dl_model(X_tensor)
            
            return output_tensor.detach().cpu().numpy()
            
        elif self.is_tf:
            return self.dl_model.predict(X)
            
        else:
            raise TypeError("Unsupported DL model type passed to adapter.")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability estimates for the input data X.
        """
        if self.is_torch:
            self.dl_model.eval()
            
            # 1. Convert numpy array to torch tensor and move to device
            X_tensor = torch.from_numpy(X).float().to(self._device)
            
            with torch.no_grad():
                # 2. Get the raw output (logits)
                output_tensor = self.dl_model(X_tensor)
            
            # 3. Apply the softmax function to get probabilities
            # NOTE: If your model already returns probabilities (e.g., has a final Softmax layer), skip this step.
            # It's generally safer to assume it returns logits and apply Softmax here.
            probabilities = torch.nn.functional.softmax(output_tensor, dim=1)
            
            # 4. Convert back to numpy array on CPU
            return probabilities.detach().cpu().numpy()
            
        elif self.is_tf:
            # TF/Keras models usually have a built-in predict_proba or just 'predict'
            # which returns probabilities if the last layer is Softmax.
            if hasattr(self.dl_model, 'predict_proba'):
                return self.dl_model.predict_proba(X)
            else:
                # For TF/Keras, 'predict' often serves as predict_proba if the
                # last layer is Softmax.
                return self.dl_model.predict(X)
                
        else:
            # Fallback for models without a recognized predict_proba interface
            raise NotImplementedError("Wrapped model does not support predict_proba.")