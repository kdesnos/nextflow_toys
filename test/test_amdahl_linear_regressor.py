import unittest
import numpy as np
from amdahl_linear_regressor import AmdahlLinearRegressor

class TestAmdahlLinearRegressor(unittest.TestCase):
    def setUp(self):
        # Simple synthetic data: y = 10 + 2*x1 + 3*x2, parallel_fraction = 0.6
        np.random.seed(42)
        self.X = np.random.randint(1, 5, size=(50, 2))
        self.nb_cores = np.random.choice([1, 2, 4, 8], size=50)
        base = 10 + 2 * self.X[:, 0] + 3 * self.X[:, 1]
        noise = np.random.normal(0, 0.05 * np.mean(base), size=base.shape)  # Add 5% noise
        self.y = base * ((1 - 0.6) + 0.6 / self.nb_cores) + noise
        self.model = AmdahlLinearRegressor()

    def test_fit_and_predict(self):
        X_full = np.column_stack([self.X, self.nb_cores])
        self.model.fit(X_full, self.y)
        y_pred = self.model.predict(X_full)
        rmse = np.sqrt(np.mean((self.y - y_pred) ** 2))
        # Print expression for debugging
        expression = self.model.get_expression(["x1", "x2", "nb_cores"])
        print(f"Model expression: {expression}")
        print(f"Parallel fraction: {self.model.parallel_fraction_}")
        print(f"RMSE: {rmse}")
        self.assertLess(rmse, 0.2 * np.mean(self.y), f"RMSE too high: {rmse}")  # Allow higher RMSE due to noise
        self.assertIsNotNone(self.model.coefficients_)
        self.assertIsNotNone(self.model.intercept_)
        self.assertIsNotNone(self.model.parallel_fraction_)
        self.assertIsNotNone(self.model.rmse_)

    def test_parallel_fraction_bounds(self):
        X_full = np.column_stack([self.X, self.nb_cores])
        self.model.fit(X_full, self.y)
        self.assertGreaterEqual(self.model.parallel_fraction_, 0)
        self.assertLessEqual(self.model.parallel_fraction_, 1)

    def test_expression_format(self):
        X_full = np.column_stack([self.X, self.nb_cores])
        self.model.fit(X_full, self.y)
        expr = self.model.get_expression(["x1", "x2", "nb_cores"])
        self.assertIn("x1", expr)
        self.assertIn("x2", expr)
        self.assertIn("nb_cores", expr)

    def test_predict_unfitted(self):
        model = AmdahlLinearRegressor()
        X_full = np.column_stack([self.X, self.nb_cores])
        with self.assertRaises(ValueError):
            model.predict(X_full)

if __name__ == "__main__":
    unittest.main()
