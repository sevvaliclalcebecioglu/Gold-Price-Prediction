from sklearn.ensemble import GradientBoostingRegressor
from src.config import RANDOM_STATE

class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

