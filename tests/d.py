import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from timerollstat import init_rolling, get_median, INT64MAX

WINDOW_SIZE_5 = 5
WINDOW_SIZE_9 = 9
WINDOW_SIZE_16 = 4
INPUT_DATA_LIST = [
    -1.0, 0, 5.0, -7.0, 10,-4, 22, 4.0,71.0, 10004.0, -0.0001,
    10.0, 20.0, 5.0, 15.0, 25.0,  8.0, 12.0, 30.0, 22.0, 18.0,
    3.0, 7.0, 28.0, 19.0, 2.0, 25.0, 10.0, 17.0, 9.0, 11.0
]
print(f"Input data: {INPUT_DATA_LIST}")
INPUT_DATA_NP = np.array(INPUT_DATA_LIST, dtype=np.float64)

def calculate_expected_rolling_median(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculates the rolling median with min_periods=1 behavior.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    if window < 1:
        raise ValueError("Window size must be at least 1.")

    n = len(data)
    expected_medians = np.empty(n, dtype=np.float64)

    for i in range(n):
        start_index = max(0, i - window + 1)
        current_window_data = data[start_index : i + 1]
        if len(current_window_data) > 0: # Should always be true with min_periods=1 logic
            expected_medians[i] = np.median(current_window_data)
    return expected_medians


EXPECTED_MEDIANS_16 = calculate_expected_rolling_median(INPUT_DATA_NP, WINDOW_SIZE_16)



rol_state = init_rolling(window_size=WINDOW_SIZE_9)

actual_medians_list = []

for value in INPUT_DATA_NP:
    median = get_median(rol_state, value)
    print()
    actual_medians_list.append(median)



