import pytest
import numpy as np
# Assuming your module is named 'timerollstat_avg' or similar
# and contains init_rolling2 and get_rolling_average_t
from timerollstat import init_rolling2, get_rolling_average_t 

# --- Test Data (can be the same or adjusted for average testing) ---
INPUT_DATA_NP = np.array([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
], dtype=np.float64)

N_POINTS = len(INPUT_DATA_NP)

# --- Generate Timestamps (consistent with your example) ---
np.random.seed(42) 
start_datetime = np.datetime64('2022-01-01T00:00:01')
timestamps_np = np.empty(N_POINTS, dtype='datetime64[ns]')
timestamps_np[0] = start_datetime
for i in range(1, N_POINTS):
    pause_seconds = np.random.randint(1, 3) # Shorter pauses for average might be fine
    if i == 5: 
        pause_seconds = 7 # A slightly longer pause
    timestamps_np[i] = timestamps_np[i-1] + np.timedelta64(pause_seconds, 's')
timestamps_ns_int64 = timestamps_np.astype(np.int64)

# --- Parameters for the test ---
TIME_WINDOWS_SECONDS_TO_TEST_AVG = [1, 2, 3, 5, 8, 10] # Adjusted for average
BIG_WINDOW_SIZE_COUNT_AVG = 1000 # Large count-based window

# --- Helper function to calculate expected true averages ---
def calculate_true_averages_for_time_window(
    input_data: np.ndarray,
    timestamps: np.ndarray,
    time_window_limit_td: np.timedelta64
) -> np.ndarray:
    """
    Calculates the true averages based on the time window logic.
    """
    n = len(input_data)
    true_averages_calculated = np.empty(n, dtype=np.float64)
    true_averages_calculated.fill(np.nan) 

    for i in range(n):
        current_timestamp = timestamps[i]
        window_elements = []
        for j in range(i, -1, -1): 
            time_diff = current_timestamp - timestamps[j]
            if time_diff <= time_window_limit_td:
                window_elements.append(input_data[j])
            else:
                break 
        
        if window_elements: # If the window is not empty
            true_averages_calculated[i] = np.mean(np.array(window_elements))
            
    return true_averages_calculated

# --- Parameterized Test for Time-Windowed Average ---
@pytest.mark.parametrize("time_window_s", TIME_WINDOWS_SECONDS_TO_TEST_AVG)
def test_rolling_average_t_fixed_data(time_window_s: int):
    """
    Tests get_rolling_average_t with predefined data for various time windows.
    """
    # 1. Arrange
    current_time_window_td = np.timedelta64(time_window_s, 's')
    current_time_window_ns_int = current_time_window_td.astype('timedelta64[ns]').astype(np.int64)

    true_averages = calculate_true_averages_for_time_window(
        INPUT_DATA_NP,
        timestamps_np,
        current_time_window_td
    )

    # Initialize rolling average state
    # Assuming init_rolling2 is the correct initializer for the average calculator
    rol_state_time_avg = init_rolling2(
        window_size=BIG_WINDOW_SIZE_COUNT_AVG,
        window_time=current_time_window_ns_int
    )
    
    test_averages_list = []

    # 2. Act
    for i in range(N_POINTS):
        avg_val = get_rolling_average_t(rol_state_time_avg, INPUT_DATA_NP[i], timestamps_ns_int64[i])
        test_averages_list.append(avg_val)
    
    test_averages_np = np.array(test_averages_list, dtype=np.float64)

    # 3. Assert (and detailed print on failure)
    are_close = np.allclose(test_averages_np, true_averages, equal_nan=True, atol=1e-5, rtol=1e-5)

    if not are_close:
        print(f"\n\n--- Detailed Comparison (Time Window: {time_window_s}s) ---")
        
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

        header = f"{'Jump':>5s} | {'Time':>7s} | {'InputVal':>10s} | {'WinSize':>7s} | {'TrueAvg':>10s} | {'TestAvg':>10s} | {'Err':>3s}"
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
                
                true_avg_val = true_averages[i]
                true_avg_str = f"{true_avg_val:10.4f}" if not np.isnan(true_avg_val) else f"{'NaN':>10s}"
                
                test_avg_val = test_averages_np[i]
                test_avg_str = f"{test_avg_val:10.4f}" if not np.isnan(test_avg_val) else f"{'NaN':>10s}"
                
                is_row_mismatch = not np.isclose(true_avg_val, test_avg_val, equal_nan=True, atol=1e-5, rtol=1e-5)
                error_marker_str = "*" if is_row_mismatch else " "
                
                print(f"{dt_str:>5s} | {ts_str:>7s} | {input_val_str} | {current_calc_win_size_str} | {true_avg_str} | {test_avg_str} | {error_marker_str:>3s}")
            
            if chunk_end_idx < N_POINTS:
                print("\n\n") # Add space between chunks for readability
        
    assert are_close, f"Test failed for rolling average with time_window_s={time_window_s}s. See detailed printout above."