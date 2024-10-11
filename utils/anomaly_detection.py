from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)

    def detect(self, data):
        return self.model.fit_predict(data)