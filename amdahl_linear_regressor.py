from scipy.optimize import minimize
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class AmdahlLinearRegressor(BaseEstimator, RegressorMixin):
    """Linear Regression with Amdahl's Law scaling for parallel execution."""
    
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self.parallel_fraction_ = None
        self.rmse_ = None
        
    def fit(self, X, y, nb_cores):
        """
        Fit the model using both linear regression parameters and parallelization factor.
        
        Parameters:
        X - Feature matrix (parameters)
        y - Actual execution times
        nb_cores - Array of core counts for each execution
        """
        n_features = X.shape[1]
        
        def predict_internal(params):
            intercept = params[0]
            coeffs = params[1:n_features+1]
            X_parallel = params[-1]
            base_times = intercept + np.sum(X * coeffs, axis=1)
            predicted_times = base_times * ((1 - X_parallel) + X_parallel / nb_cores)
            return predicted_times
        
        def loss_function(params):
            predicted = predict_internal(params)
            return np.sqrt(np.mean((predicted - y)**2))  # RMSE
        
        initial_params = np.zeros(n_features + 2)  # +1 for intercept, +1 for parallel_fraction
        initial_params[-1] = 0.5  # Initial guess for parallel_fraction
        bounds = [(None, None)] * (n_features + 1)
        bounds.append((0, 1))
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B')
        self.intercept_ = result.x[0]
        self.coefficients_ = result.x[1:n_features+1]
        self.parallel_fraction_ = result.x[-1]
        self.rmse_ = result.fun  # Final RMSE
        return self
    
    def predict(self, X, nb_cores=1):
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")
        base_times = self.intercept_ + np.sum(X * self.coefficients_, axis=1)
        predicted_times = base_times * ((1 - self.parallel_fraction_) + self.parallel_fraction_ / nb_cores)
        return predicted_times

    def get_expression(self, param_names):
        terms = [f"{self.intercept_:.2f}"]
        for i, coef in enumerate(self.coefficients_):
            if coef != 0:
                terms.append(f"({coef:.2f} * {param_names[i]})")
        base_expr = " + ".join(terms)
        full_expr = f"({base_expr}) * ({1-self.parallel_fraction_:.4f} + {self.parallel_fraction_:.4f}/cores)"
        return full_expr
