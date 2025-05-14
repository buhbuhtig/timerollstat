import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from timerollstat import init_rolling, get_quantile

# --- Test Parameters ---
WINDOW_SIZES = [5, 9, 4] # window sizes
QUANTILES_TO_TEST = [0.2, 0.4, 0.65, 0.9]

INPUT_DATA_LIST = [
    -1.0, 0.0, 5.0, -7.0, 10.0, -4.0, 22.0, 4.0, 71.0, 10004.0, -0.0001,
    10.0, 20.0, 5.0, 15.0, 25.0,  8.0, 12.0, 30.0, 22.0, 18.0,
    3.0, 7.0, 28.0, 19.0, 2.0, 25.0, 10.0, 17.0, 9.0, 11.0
]
INPUT_DATA_NP = np.array(INPUT_DATA_LIST, dtype=np.float64)

# print(f"Input data: {INPUT_DATA_NP}") # For debugging

# --- Function to calculate expected rolling quantiles ---
def calculate_expected_rolling_quantile(data: np.ndarray, window: int, quantile_val: float) -> np.ndarray:
    """
    Calculates the rolling quantile with min_periods=1 behavior using NumPy.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    if window < 1:
        raise ValueError("Window size must be at least 1.")
    if not (0.0 <= quantile_val <= 1.0):
        raise ValueError("Quantile value must be between 0.0 and 1.0.")

    n = len(data)
    expected_quantiles = np.empty(n, dtype=np.float64)

    for i in range(n):
        start_index = max(0, i - window + 1)
        current_window_data = data[start_index : i + 1]
        if len(current_window_data) > 0:
            expected_quantiles[i] = np.quantile(current_window_data, quantile_val, method='linear')
    return expected_quantiles

window_size =5
quantile_val = 0.2
expected_quantiles_np = calculate_expected_rolling_quantile(INPUT_DATA_NP, window_size, quantile_val)
rol_state = init_rolling(window_size=window_size, q=quantile_val)
actual_quantiles_list = []


for value in INPUT_DATA_NP:
    q_result = get_quantile(rol_state, value) # Using get_quantile
    actual_quantiles_list.append(q_result)

actual_quantiles_np = np.array(actual_quantiles_list, dtype=np.float64)


ic = np.isclose(expected_quantiles_np, actual_quantiles_np, rtol=1e-5, atol=1e-8, equal_nan=True)
print(INPUT_DATA_NP)

print(expected_quantiles_np)
print(actual_quantiles_np)
print(all(ic))
assert_array_almost_equal(actual_quantiles_np, expected_quantiles_np, decimal=5,
                              err_msg='ddddddddd')