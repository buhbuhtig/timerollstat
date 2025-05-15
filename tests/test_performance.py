import numpy as np
import pandas as pd
import time

# Your actual imports
from timerollstat import rolling_median, rolling_quantile, rolling_median_t, rolling_quantile_t 

# --- Common Parameters ---
array_size = 1_500_000 
window_k_count_based = 5_000 
np.random.seed(42) 
random_numpy_array = np.random.rand(array_size).astype(np.float64)

print(f"Performance Benchmark")
print(f"---------------------")
print(f"Array size for all tests: {array_size:,}") 
print(f"Pandas version: {pd.__version__}") # Version info before tables
print(f"NumPy version: {np.__version__}")

all_benchmark_entries = [] # One list to hold all entries for structured printing

# --- Helper function to print Markdown table (MODIFIED) ---
def print_markdown_table_paired(title, results_list, fixed_params_info=""):
    if not results_list:
        print(f"\n\n--- {title} ---")
        print(fixed_params_info)
        print("No results to display.")
        return

    print(f"\n\n--- {title} ---")
    if fixed_params_info:
        print(fixed_params_info)
    
    col_method = "Method"
    col_details = "Window / Details" 
    col_time = "Time (s)"
    col_speedup = "Speedup (vs Pandas Batch)" # Clarified

    # Calculate max widths
    # Ensure all keys exist using .get() with a default
    w_method = max(len(col_method), max(len(str(r.get("Method", ""))) for r in results_list)) + 2
    w_details = max(len(col_details), max(len(str(r.get("Window / Details", ""))) for r in results_list)) + 2
    w_time = max(len(col_time), max(len(str(r.get("Time (s)", ""))) for r in results_list)) + 2
    w_speedup = max(len(col_speedup), max(len(str(r.get("Speedup (vs Pandas Batch)", ""))) for r in results_list)) + 2
    
    header_str = f"| {col_method:<{w_method-2}} | {col_details:<{w_details-2}} | {col_time:>{w_time-2}} | {col_speedup:>{w_speedup-2}} |"
    separator_str = f"|{'-' * w_method}|{'-' * w_details}|{'-' * w_time}|{'-' * w_speedup}|"

    print(header_str)
    print(separator_str)

    for result in results_list:
        speedup_val = result.get("Speedup (vs Pandas Batch)", "") # Get value, or empty if not present
        row_str = (
            f"| {str(result.get('Method', 'N/A')):<{w_method-2}} "
            f"| {str(result.get('Window / Details', 'N/A')):<{w_details-2}} " 
            f"| {str(result.get('Time (s)', 'N/A')):>{w_time-2}} "
            f"| {str(speedup_val):>{w_speedup-2}} |" # Display speedup or empty
        )
        print(row_str)

# ==============================================================================
# COUNT-BASED TESTS
# ==============================================================================
current_test_block_title = f"Benchmark: Count-Based Windows (W_count={window_k_count_based:,})"
current_test_block_results = []

# --- rolling_median (count-based) ---
warmup_array_median_cb = np.random.rand(100).astype(np.float64)
_ = rolling_median(warmup_array_median_cb, 5)
start_time_numba_median_cb = time.perf_counter()
_ = rolling_median(random_numpy_array, window_k_count_based)
numba_median_cb_time = time.perf_counter() - start_time_numba_median_cb
pd_series_median_cb = pd.Series(random_numpy_array)
start_time_pandas_median_cb = time.perf_counter()
_ = pd_series_median_cb.rolling(window=window_k_count_based, min_periods=1).median()
pandas_median_cb_time = time.perf_counter() - start_time_pandas_median_cb
speedup_median_cb = (pandas_median_cb_time / numba_median_cb_time) if numba_median_cb_time > 0 and pandas_median_cb_time > 0 else ""

current_test_block_results.append({
    "Method": "timerollstat_median (online)", "Window / Details": "q=0.5",
    "Time (s)": f"{numba_median_cb_time:.4f}", 
    "Speedup (vs Pandas Batch)": f"{speedup_median_cb:.2f}x" if isinstance(speedup_median_cb, float) else "N/A"
})
current_test_block_results.append({
    "Method": "Pandas (batch)", "Window / Details": "q=0.5",
    "Time (s)": f"{pandas_median_cb_time:.4f}", 
    "Speedup (vs Pandas Batch)": "" # Empty for Pandas row
})

# --- rolling_quantile (count-based) ---
quantiles_to_test_cb = [0.1, 0.25, 0.5, 0.75, 0.9] 
warmup_array_quantile_cb = np.random.rand(100).astype(np.float64)
_ = rolling_quantile(warmup_array_quantile_cb, 5, 0.5)
for q_val_cb in quantiles_to_test_cb:
    details_str_cb = f"q={q_val_cb}"
    start_time_numba_quantile_cb = time.perf_counter()
    _ = rolling_quantile(random_numpy_array, window_k_count_based, q_val_cb)
    numba_quantile_cb_time = time.perf_counter() - start_time_numba_quantile_cb
    pd_series_quantile_cb = pd.Series(random_numpy_array) 
    start_time_pandas_quantile_cb = time.perf_counter()
    _ = pd_series_quantile_cb.rolling(window=window_k_count_based, min_periods=1).quantile(q_val_cb)
    pandas_quantile_cb_time = time.perf_counter() - start_time_pandas_quantile_cb
    speedup_quantile_cb = (pandas_quantile_cb_time / numba_quantile_cb_time) if numba_quantile_cb_time > 0 and pandas_quantile_cb_time > 0 else ""
    
    current_test_block_results.append({
        "Method": "timerollstat_quantile (online)", "Window / Details": details_str_cb,
        "Time (s)": f"{numba_quantile_cb_time:.4f}", 
        "Speedup (vs Pandas Batch)": f"{speedup_quantile_cb:.2f}x" if isinstance(speedup_quantile_cb, float) else "N/A"
    })
    current_test_block_results.append({
        "Method": "Pandas (batch)", "Window / Details": details_str_cb,
        "Time (s)": f"{pandas_quantile_cb_time:.4f}", 
        "Speedup (vs Pandas Batch)": ""
    })
print_markdown_table_paired(current_test_block_title, current_test_block_results)


# ==============================================================================
# TIME-BASED TESTS SETUP
# ==============================================================================
N_POINTS_TIME_TEST = array_size 
np.random.seed(43) 
start_datetime = np.datetime64('2023-01-01T00:00:00')
timestamps_datetime64_ns = np.empty(N_POINTS_TIME_TEST, dtype='datetime64[ns]')
timestamps_datetime64_ns[0] = start_datetime
for i in range(1, N_POINTS_TIME_TEST):
    pause_seconds = np.random.randint(1, 11) 
    timestamps_datetime64_ns[i] = timestamps_datetime64_ns[i-1] + np.timedelta64(pause_seconds, 's')
timestamps_int64_ns = timestamps_datetime64_ns.astype(np.int64)

time_windows_config = {
    "5min": np.timedelta64(5, 'm'), "30min": np.timedelta64(30, 'm'), 
    "1h": np.timedelta64(1, 'h'), "24h": np.timedelta64(24, 'h')
}
MAX_COUNT_WINDOW_FOR_TIME_TEST = N_POINTS_TIME_TEST 

# ==============================================================================
# TIME-BASED ROLLING_MEDIAN_T
# ==============================================================================
current_test_block_title_median_t = "Benchmark: Time-Based Rolling Median"
current_test_block_results_median_t = []

warmup_array_median_t = np.random.rand(100).astype(np.float64)
warmup_timestamps_median_t_ns = timestamps_int64_ns[:100] 
warmup_time_window_median_t_ns = np.timedelta64(1, 's').astype('timedelta64[ns]').astype(np.int64)
_ = rolling_median_t(warmup_array_median_t, warmup_timestamps_median_t_ns, 
                     warmup_time_window_median_t_ns, 100) 

for pd_win_str_mt, time_window_td_mt in time_windows_config.items():
    details_str_mt = f"q=0.5, {pd_win_str_mt}"
    time_window_ns_for_numba_mt = time_window_td_mt.astype('timedelta64[ns]').astype(np.int64)
    start_time_numba_mt = time.perf_counter()
    _ = rolling_median_t(random_numpy_array, timestamps_int64_ns, 
                         time_window_ns_for_numba_mt, MAX_COUNT_WINDOW_FOR_TIME_TEST)
    numba_mt_time = time.perf_counter() - start_time_numba_mt
    pd_series_time_indexed_mt = pd.Series(random_numpy_array, index=pd.to_datetime(timestamps_datetime64_ns))
    start_time_pandas_mt = time.perf_counter()
    _ = pd_series_time_indexed_mt.rolling(window=pd_win_str_mt, min_periods=1).median()
    pandas_mt_time = time.perf_counter() - start_time_pandas_mt
    speedup_mt = (pandas_mt_time / numba_mt_time) if numba_mt_time > 0 and pandas_mt_time > 0 else ""
    
    current_test_block_results_median_t.append({
        "Method": "timerollstat_median_t (online)", "Window / Details": details_str_mt,
        "Time (s)": f"{numba_mt_time:.4f}", 
        "Speedup (vs Pandas Batch)": f"{speedup_mt:.2f}x" if isinstance(speedup_mt, float) else "N/A"
    })
    current_test_block_results_median_t.append({
        "Method": "Pandas (batch)", "Window / Details": details_str_mt,
        "Time (s)": f"{pandas_mt_time:.4f}", 
        "Speedup (vs Pandas Batch)": ""
    })
print_markdown_table_paired(current_test_block_title_median_t, current_test_block_results_median_t,
                            f"Max count window: {MAX_COUNT_WINDOW_FOR_TIME_TEST:,}. Pandas operates in batch mode.")


# ==============================================================================
# TIME-BASED ROLLING_QUANTILE_T (various q, excluding 0.5)
# ==============================================================================
quantiles_to_test_time = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9] 
warmup_array_quantile_t = np.random.rand(100).astype(np.float64)
warmup_timestamps_quantile_t_ns = timestamps_int64_ns[:100]
warmup_time_window_quantile_t_ns = np.timedelta64(1, 's').astype('timedelta64[ns]').astype(np.int64)
_ = rolling_quantile_t(warmup_array_quantile_t, warmup_timestamps_quantile_t_ns, 
                       warmup_time_window_quantile_t_ns, 100, 0.5) 

for q_val_time in quantiles_to_test_time:
    current_test_block_title_quantile_t = f"Benchmark: Time-Based Rolling Quantile (q={q_val_time})"
    current_test_block_results_quantile_t = []
    for pd_win_str_qt, time_window_td_qt in time_windows_config.items():
        details_str_qt = f"W_time={pd_win_str_qt}"
        time_window_ns_for_numba_qt = time_window_td_qt.astype('timedelta64[ns]').astype(np.int64)
        start_time_numba_qt = time.perf_counter()
        _ = rolling_quantile_t(random_numpy_array, timestamps_int64_ns, 
                               time_window_ns_for_numba_qt, MAX_COUNT_WINDOW_FOR_TIME_TEST,
                               q_val_time)
        numba_qt_time = time.perf_counter() - start_time_numba_qt
        pd_series_time_indexed_qt = pd.Series(random_numpy_array, index=pd.to_datetime(timestamps_datetime64_ns))
        start_time_pandas_qt = time.perf_counter()
        _ = pd_series_time_indexed_qt.rolling(window=pd_win_str_qt, min_periods=1).quantile(q_val_time)
        pandas_qt_time = time.perf_counter() - start_time_pandas_qt
        speedup_qt = (pandas_qt_time / numba_qt_time) if numba_qt_time > 0 and pandas_qt_time > 0 else ""
        
        current_test_block_results_quantile_t.append({
            "Method": "timerollstat_quantile_t (online)", "Window / Details": details_str_qt,
            "Time (s)": f"{numba_qt_time:.4f}", 
            "Speedup (vs Pandas Batch)": f"{speedup_qt:.2f}x" if isinstance(speedup_qt, float) else "N/A"
        })
        current_test_block_results_quantile_t.append({
            "Method": "Pandas (batch)", "Window / Details": details_str_qt,
            "Time (s)": f"{pandas_qt_time:.4f}", 
            "Speedup (vs Pandas Batch)": ""
        })
    print_markdown_table_paired(current_test_block_title_quantile_t, current_test_block_results_quantile_t,
                                f"Max count window: {MAX_COUNT_WINDOW_FOR_TIME_TEST:,}. Pandas operates in batch mode.")