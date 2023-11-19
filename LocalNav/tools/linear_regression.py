import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression(data):
    # Separate x and y
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    # Filter out rows with infinite values
    mask_x = ~np.isinf(X)
    mask_y = ~np.isinf(y)
    mask = mask_x & mask_y # combine both mask
    X_filtered = X[mask].reshape(-1, 1)
    y_filtered = y[mask].reshape(-1, 1)

    if any(mask):
        # Create a linear regression model
        model = LinearRegression()
        # Fit the model to the data
        model.fit(X_filtered, y_filtered)
        # Make predictions
        y_pred = model.predict(X_filtered)
    
    # Put back missing data after linear regression
    i=0
    j=0
    while i < len(data):
        if not mask[i]:
            X[i] = np.nan
            y[i] = np.nan
            i = i + 1
        elif mask[i]:
            y[i] = y_pred[j]
            i = i + 1
            j = j + 1
    
    return X, y, mask