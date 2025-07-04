import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(dict)
    
    def fit(self, X, y):
        # Calculate class probabilities
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        self.class_probs = {c: count/total_samples for c, count in zip(classes, counts)}
        
        # Calculate feature probabilities for each class
        for cls in classes:
            cls_samples = X[y == cls]
            for feature_idx in range(X.shape[1]):
                feature_values, feature_counts = np.unique(cls_samples[:, feature_idx], return_counts=True)
                total_cls_samples = len(cls_samples)
                self.feature_probs[cls][feature_idx] = {
                    val: count/total_cls_samples for val, count in zip(feature_values, feature_counts)
                }
    
    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = -1
            best_class = None
            for cls in self.class_probs:
                prob = self.class_probs[cls]
                for feature_idx, feature_val in enumerate(sample):
                    if feature_val in self.feature_probs[cls][feature_idx]:
                        prob *= self.feature_probs[cls][feature_idx][feature_val]
                    else:
                        prob = 0  # if feature value wasn't seen in training
                        break
                if prob > max_prob:
                    max_prob = prob
                    best_class = cls
            predictions.append(best_class)
        return np.array(predictions)

# Example usage:
if __name__ == "__main__":
    # Sample data: Outlook (0=Sunny, 1=Rainy), Temp (0=Hot, 1=Mild), PlayTennis (0=No, 1=Yes)
    X = np.array([[0, 0], [0, 0], [1, 0], [1, 1], [1, 1]])
    y = np.array([0, 0, 1, 1, 1])
    
    nb = NaiveBayes()
    nb.fit(X, y)
    print("Predictions:", nb.predict(np.array([[1, 0], [0, 1]])))