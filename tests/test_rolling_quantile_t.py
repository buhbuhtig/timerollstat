import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

# Assuming INT64MAX and your functions are available through this import
from timerollstat import init_rolling, get_quantile_t, INT64MAX # Using get_quantile_t

# --- Test Data (remains the same) ---
INPUT_DATA_NP = np.array([
    -1.0, 0.0, 5.0, -7.0, 10.0, -4.0, 22.0, 4.0, 71.0, 10004.0, -0.0001,
    10.0, 20.0, 5.0, 15.0, 25.0,  8.0, 12.0, 30.0, 22.0, 18.0,
    3.0, 7.0, 28.0, 19.0, 2.0, 25.0, 10.0, 17.0, 9.0, 11.0
], dtype=np.float64)

N_POINTS = len(INPUT_DATA_NP)

# --- Generate Timestamps (remains the same, defined once for all tests) ---
np.random.seed(42) # Ensure timestamps are the same for each test run
start_datetime = np.datetime64('2022-01-01T00:00:01')
timestamps_np = np.empty(N_POINTS, dtype='datetime64[ns]')
timestamps_np[0] = start_datetime
for i in range(1, N_POINTS):
    pause_seconds = np.random.randint(1, 4) # Pauses of 1, 2, or 3 seconds
    if i == 10: # Special pause for one data point
        pause_seconds = 10
    timestamps_np[i] = timestamps_np[i-1] + np.timedelta64(pause_seconds, 's')
timestamps_ns_int64 = timestamps_np.astype(np.int64)

# --- Parameters for the test (parametrized) ---
TIME_WINDOWS_SECONDS_TO_TEST = [1, 2, 4, 5, 10, 11]
QUANTILES_TO_TEST = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9]
BIG_WINDOW_SIZE_COUNT = 10000 # Large count-based window, time window should be dominant

# --- Helper function to calculate expected true quantiles ---
def calculate_true_quantiles_for_time_window(
    input_data: np.ndarray,
    timestamps: np.ndarray,
    time_window_limit_td: np.timedelta64,
    quantile_value: float
) -> np.ndarray:
    """
    Calculates the true quantiles based on the time window logic.
    The count-based window is assumed to be very large and not limiting here.
    Uses np.quantile for the calculation.
    """
    n = len(input_data)
    elements_in_time_window_count = np.empty(n, dtype=np.int64)
    true_quantiles_calculated = np.empty(n, dtype=np.float64)
    true_quantiles_calculated.fill(np.nan) # Initialize with NaN

    for i in range(n):
        current_timestamp = timestamps[i]
        effective_window_count = 0
        for j in range(i, -1, -1): # Iterate backwards from current point
            time_diff = current_timestamp - timestamps[j]
            if time_diff <= time_window_limit_td:
                effective_window_count = i - j + 1
            else:
                break # This element (and earlier ones) are outside the time window
        elements_in_time_window_count[i] = effective_window_count
    
    for i in range(n):
        window_len = elements_in_time_window_count[i]
        if window_len > 0:
            current_slice = input_data[i - window_len + 1 : i + 1]
            true_quantiles_calculated[i] = np.quantile(current_slice, quantile_value) # Use np.quantile
            
    return true_quantiles_calculated

# --- Parameterized Test for Time-Windowed Quantile ---
@pytest.mark.parametrize("time_window_s", TIME_WINDOWS_SECONDS_TO_TEST)
@pytest.mark.parametrize("quantile_value", QUANTILES_TO_TEST)
def test_rolling_quantile_t_fixed_data(time_window_s: int, quantile_value: float):
    """
    Tests get_quantile_t with predefined data for various time windows and quantiles.
    """
    # 1. Arrange
    current_time_window_td = np.timedelta64(time_window_s, 's')
    current_time_window_ns_int = current_time_window_td.astype('timedelta64[ns]').astype(np.int64)

    true_quantiles_np = calculate_true_quantiles_for_time_window(
        INPUT_DATA_NP,
        timestamps_np,
        current_time_window_td,
        quantile_value # Pass the current quantile value
    )

    # print(f"\nTesting Time Window: {time_window_s}s, Quantile: {quantile_value}") # For debugging
    # print(f"Expected Quantiles (first 10): {true_quantiles_np[:10]}") # For debugging

    
    rol_state_time = init_rolling(
        window_size=BIG_WINDOW_SIZE_COUNT,
        window_time=current_time_window_ns_int,
        q=quantile_value # Pass the current quantile value
    )


    test_quantiles_list = []

    # 2. Act
    for i in range(N_POINTS):
        # Use get_quantile_t
        quantile_val_calculated = get_quantile_t(rol_state_time, INPUT_DATA_NP[i], timestamps_ns_int64[i])
        test_quantiles_list.append(quantile_val_calculated)
    
    test_quantiles_np = np.array(test_quantiles_list, dtype=np.float64)
    # print(f"Actual Quantiles (first 10):   {test_quantiles_np[:10]}") # For debugging

    # 3. Assert (and detailed print on failure)
    are_close = np.allclose(test_quantiles_np, true_quantiles_np, equal_nan=True, atol=1e-5)

    if not are_close:
        print(f"\n\n--- Detailed Comparison (Time Window: {time_window_s}s, Quantile: {quantile_value}) ---")
        
        time_deltas_s = np.zeros(N_POINTS, dtype=np.int64)
        if N_POINTS > 1:
            time_deltas_s[1:] = (timestamps_np[1:] - timestamps_np[:-1]).astype('timedelta64[s]').astype(np.int64)

        effective_window_sizes_for_print = np.empty(N_POINTS, dtype=np.int64)
        for i in range(N_POINTS):
            count = 0
            for j in range(i, -1, -1):
                if (timestamps_np[i] - timestamps_np[j]) <= current_time_window_td:
                    count +=1
                else:
                    break
            effective_window_sizes_for_print[i] = count

        # Updated header for quantiles
        header = f"{'Jump':>5s} | {'Time':>7s} | {'InputVal':>10s} | {'WinSize':>7s} | {'TrueQuant':>10s} | {'TestQuant':>10s} | {'Err':>3s}"
        separator = "-" * len(header)
        
        ITEMS_PER_CHUNK = 15 

        for chunk_start_idx in range(0, N_POINTS, ITEMS_PER_CHUNK):
            chunk_end_idx = min(chunk_start_idx + ITEMS_PER_CHUNK, N_POINTS)
            
            print(header)
            print(separator)
            
            for i in range(chunk_start_idx, chunk_end_idx):
                dt_str = f"{time_deltas_s[i]:+}s" if i > 0 else "0s"
                
                ts_total_seconds = timestamps_np[i].astype('datetime64[s]').astype(np.int64)
                ts_minutes = (ts_total_seconds // 60) % 60
                ts_seconds = ts_total_seconds % 60
                ts_str = f"{ts_minutes:02d}:{ts_seconds:02d}"
                
                input_val_str = f"{INPUT_DATA_NP[i]:10.4f}"
                
                current_calc_win_size_str = f"{effective_window_sizes_for_print[i]:7d}"
                
                true_quantile_val = true_quantiles_np[i] # Use true_quantiles_np
                true_quantile_str = f"{true_quantile_val:10.4f}" if not np.isnan(true_quantile_val) else f"{'NaN':>10s}"
                
                test_quantile_val = test_quantiles_np[i] # Use test_quantiles_np
                test_quantile_str = f"{test_quantile_val:10.4f}" if not np.isnan(test_quantile_val) else f"{'NaN':>10s}"
                
                is_row_mismatch = not np.isclose(true_quantile_val, test_quantile_val, equal_nan=True, atol=1e-5)
                error_marker_str = "*" if is_row_mismatch else " "
                
                print(f"{dt_str:>5s} | {ts_str:>7s} | {input_val_str} | {current_calc_win_size_str} | {true_quantile_str} | {test_quantile_str} | {error_marker_str:>3s}")
            
            if chunk_end_idx < N_POINTS:
                print("\n\n")
        
    assert are_close, f"Test failed for time_window_s={time_window_s}s and quantile={quantile_value}. See detailed printout above."