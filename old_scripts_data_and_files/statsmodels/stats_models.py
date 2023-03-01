import numpy as np

import statsmodels.api as sm

# Generate artificial data (2 regressors + constant)
nobs = 10

X = np.random.random((nobs, 2))

X = sm.add_constant(X)

beta = [1, .1, .5]

e = np.random.random(nobs)

y = np.dot(X, beta) + e

print("X = ", X)
print("Y = ", y)

# Fit regression model
ols = sm.OLS(y, X)
resulted_ols_model = ols.fit()
y_predicted = resulted_ols_model.predict(X)

print("y_predicted", y_predicted)

# Inspect the results
print(resulted_ols_model.summary())