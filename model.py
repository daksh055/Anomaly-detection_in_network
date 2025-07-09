from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
    
    def train(self, X):
        """Train the model"""
        self.model.fit(X)
    
    def predict(self, X):
        """Predict anomalies (1=normal, -1=anomaly)"""
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """Compare with actual labels"""
        preds = self.predict(X)
        # Convert (-1,1) to (1,0) to match true labels
        preds = np.where(preds == -1, 1, 0)
        accuracy = (preds == y_true).mean()
        return accuracy