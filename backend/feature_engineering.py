
import numpy as np


def add_features(X):

    X = np.asarray(X, dtype=float)

    N = X[:, 0]
    P = X[:, 1]
    K = X[:, 2]
    temperature = X[:, 3]
    humidity = X[:, 4]
    ph = X[:, 5]
    rainfall = X[:, 6]

    total_npk = N + P + K

    nitrogen_ratio = N / (total_npk + 1)
    phosphorus_ratio = P / (total_npk + 1)
    potassium_ratio = K / (total_npk + 1)

    ph_distance = abs(ph - 6.5)

    temperature_humidity = temperature * humidity / 100

    rainfall_humidity = rainfall / (humidity + 1)

    log_rainfall = np.log1p(rainfall)

    extra_features = np.column_stack([
        total_npk,
        nitrogen_ratio,
        phosphorus_ratio,
        potassium_ratio,
        ph_distance,
        temperature_humidity,
        rainfall_humidity,
        log_rainfall
    ])

    return np.hstack([X, extra_features])
