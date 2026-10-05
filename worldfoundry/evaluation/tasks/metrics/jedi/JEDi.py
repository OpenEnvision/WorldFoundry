from .utils import model_cleanup, feature_aggregator
import os
import numpy as np

class JEDiMetric:
    def __init__(self, feature_path=None, model_dir=None, config_path=None):
        self.feature_path = feature_path
        if not feature_path:
            import warnings
            warnings.warn("feature_path is not provided, will not save computed features.")
        self.model_dir = model_dir if model_dir is not None else os.getcwd()
        self.config_path = config_path
        if self.feature_path:
            os.makedirs(self.feature_path, exist_ok=True)
    
    def compute_metric(self):
        assert hasattr(self, 'train_features'), "train_features is not loaded"
        assert hasattr(self, 'test_features'), "test_features is not loaded"
        from .mmd_polynomial import mmd_poly
        return mmd_poly(self.train_features, self.test_features, degree=2, coef0=0)*100

    def load_features(self, train_loader=None, test_loader=None, num_samples=5000):
        try:
            for split, loader in (("train", train_loader), ("test", test_loader)):
                filename = f'{self.feature_path}/{split}.npy' if self.feature_path is not None else None
                if filename is not None and os.path.exists(filename):
                    features = np.load(filename)[:num_samples]
                else:
                    assert loader is not None, f"{split}_loader is not provided"
                    if not hasattr(self, 'vjepa'):
                        from .V_JEPA import VJEPA

                        self.vjepa = VJEPA(model_dir=self.model_dir, config_fname=self.config_path)
                    print(f"Computing features for {split} set")
                    # Loader batches: (B, T, C, H, W), range [0, 1].
                    features = feature_aggregator(self.vjepa, loader, num_samples=num_samples, filename=filename)
                setattr(self, f'{split}_features', features)
        finally:
            if hasattr(self, 'vjepa'):
                del self.vjepa
                model_cleanup()
        return self.train_features, self.test_features
