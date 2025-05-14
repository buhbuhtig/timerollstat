import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

# Assuming INT64MAX and your functions are available through this import
from timerollstat import init_rolling, get_median, INT64MAX # Using get_median

# --- Test Parameters ---
# Combine window sizes into a single list for parametrization
WINDOW_SIZES_FOR_MEDIAN_TEST = [5, 9, 4] # Same as WINDOW_SIZE_5, WINDOW_SIZE_9, WINDOW_SIZE_16

INPUT_DATA_LIST = [
    -1.0, 0.0, 5.0, -7.0, 10.0, -4.0, 22.0, 4.0, 71.0, 10004.0, -0.0001,
    10.0, 20.0, 5.0, 15.0, 25.0,  8.0, 12.0, 30.0, 22.0, 18.0,
    3.0, 7.0, 28.0, 19.0, 2.0, 25.0, 10.0, 17.0, 9.0, 11.0
]
# print(f"Input data (median test): {INPUT_DATA_LIST}") # For debugging
INPUT_DATA_NP = np.array(INPUT_DATA_LIST, dtype=np.float64)

# --- Function to calculate expected rolling medians (remains the same) ---
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
    expected_medians.fill(np.nan) # Initialize with NaN

    for i in range(n):
        start_index = max(0, i - window + 1)
        current_window_data = data[start_index : i + 1]
        if len(current_window_data) > 0:
            expected_medians[i] = np.median(current_window_data)
    return expected_medians

# --- Parameterized Test for Rolling Median ---
@pytest.mark.parametrize("window_size", WINDOW_SIZES_FOR_MEDIAN_TEST)
def test_rolling_median_fixed_data(window_size: int):
    """
    Tests get_median with predefined data for various window sizes.
    """
    # 1. Arrange
    expected_medians_np = calculate_expected_rolling_median(INPUT_DATA_NP, window_size)
    # print(f"\nTesting Median W={window_size}") # For debugging

    
    rol_state = init_rolling(window_size=window_size)


    actual_medians_list = []

    # 2. Act
    for value in INPUT_DATA_NP:
        median_result = get_median(rol_state, value)
        actual_medians_list.append(median_result)

    actual_medians_np = np.array(actual_medians_list, dtype=np.float64)

    # 3. Assert
    error_message = (
        f"Calculated rolling medians do not match expected values "
        f"for window_size={window_size}."
    )
    assert_array_almost_equal(actual_medians_np, expected_medians_np, decimal=5,
                              err_msg=error_message)