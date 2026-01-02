from sklearn.ensemble import RandomForestRegressor
from src.config import RANDOM_STATE

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
