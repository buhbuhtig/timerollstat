import numpy as np
import pandas as pd
import time

# Your actual imports - ensure rolling_quantile_t is also imported
from timerollstat import rolling_median, rolling_quantile, rolling_median_t, rolling_quantile_t 
# Assuming INT64MAX is available if needed by any init functions
# from timerollstat import INT64MAX 

# --- Common Parameters ---
array_size = 1_500_000 
window_k_count_based = 5_000 

np.random.seed(42) 
random_numpy_array = np.random.rand(array_size).astype(np.float64)

print(f"Performance Benchmark")
print(f"---------------------")
print(f"Array size for all tests: {array_size:,}") 
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
# import numba
# print(f"Numba version: {numba.__version__}")

benchmark_results = [] # Initialize list to store all benchmark results

# ==============================================================================
# BLOCK FOR COUNT-BASED ROLLING_MEDIAN (q=0.5)
# (This block remains as it was)
# ==============================================================================
print(f"\n\n--- Benchmarking: rolling_median (q=0.5, count-based window: {window_k_count_based}) ---")
# ... (warmup and timing code for rolling_median vs pandas.rolling.median) ...
warmup_array_median = np.random.rand(100).astype(np.float64) # Placeholder for brevity
warmup_window_median = 5
_ = rolling_median(warmup_array_median, warmup_window_median)
start_time_numba_median = time.perf_counter()
_ = rolling_median(random_numpy_array, window_k_count_based)
end_time_numba_median = time.perf_counter()
numba_median_time = end_time_numba_median - start_time_numba_median
pandas_data_series_median = pd.Series(random_numpy_array)
start_time_pandas_median = time.perf_counter()
_ = pandas_data_series_median.rolling(window=window_k_count_based, min_periods=1).median()
end_time_pandas_median = time.perf_counter()
pandas_median_time = end_time_pandas_median - start_time_pandas_median
if numba_median_time > 0 and pandas_median_time > 0:
    speedup_median = pandas_median_time / numba_median_time
else:
    speedup_median = np.nan 
benchmark_results.append({
    "Function": "rolling_median (count)",
    "Quantile/Window": f"q=0.5, W={window_k_count_based}",
    "timerollstat (s)": f"{numba_median_time:.4f}",
    "Pandas (s)": f"{pandas_median_time:.4f}",
    "Speedup": f"{speedup_median:.2f}x" if not np.isnan(speedup_median) else "N/A"
})


# ==============================================================================
# LOOP FOR COUNT-BASED ROLLING_QUANTILE (various q)
# (This block remains as it was)
# ==============================================================================
print(f"\n\n--- Benchmarking: rolling_quantile (various q, count-based window: {window_k_count_based}) ---")
quantiles_to_test_count_based = [0.1, 0.25, 0.5, 0.75, 0.9]
# ... (warmup and timing code for rolling_quantile vs pandas.rolling.quantile) ...
warmup_array_quantile = np.random.rand(100).astype(np.float64) # Placeholder
warmup_window_quantile = 5
_ = rolling_quantile(warmup_array_quantile, warmup_window_quantile, 0.5)
for q_val_count in quantiles_to_test_count_based: # Renamed q_val to avoid conflict
    start_time_numba_quantile = time.perf_counter()
    _ = rolling_quantile(random_numpy_array, window_k_count_based, q_val_count)
    end_time_numba_quantile = time.perf_counter()
    numba_quantile_time = end_time_numba_quantile - start_time_numba_quantile
    pandas_data_series_quantile = pd.Series(random_numpy_array)
    start_time_pandas_quantile = time.perf_counter()
    _ = pandas_data_series_quantile.rolling(window=window_k_count_based, min_periods=1).quantile(q_val_count)
    end_time_pandas_quantile = time.perf_counter()
    pandas_quantile_time = end_time_pandas_quantile - start_time_pandas_quantile
    if numba_quantile_time > 0 and pandas_quantile_time > 0:
        speedup_quantile = pandas_quantile_time / numba_quantile_time
    else:
        speedup_quantile = np.nan
    benchmark_results.append({
        "Function": "rolling_quantile (count)",
        "Quantile/Window": f"q={q_val_count}, W={window_k_count_based}",
        "timerollstat (s)": f"{numba_quantile_time:.4f}",
        "Pandas (s)": f"{pandas_quantile_time:.4f}",
        "Speedup": f"{speedup_quantile:.2f}x" if not np.isnan(speedup_quantile) else "N/A"
    })


# ==============================================================================
# BLOCK FOR TIME-BASED ROLLING_MEDIAN_T (q=0.5)
# (This block remains as it was, using rolling_median_t for q=0.5 specifically)
# ==============================================================================
print("\n\n--- Benchmarking: rolling_median_t (time-based window, q=0.5) ---")
print("This section benchmarks rolling median calculations based on a fixed time duration window.")

N_POINTS_TIME_TEST_median = array_size 
np.random.seed(43) 
start_datetime_median_t = np.datetime64('2023-01-01T00:00:00') # Use different start if needed
timestamps_datetime64_ns_median_t = np.empty(N_POINTS_TIME_TEST_median, dtype='datetime64[ns]')
timestamps_datetime64_ns_median_t[0] = start_datetime_median_t
for i in range(1, N_POINTS_TIME_TEST_median):
    pause_seconds = np.random.randint(1, 11) 
    timestamps_datetime64_ns_median_t[i] = timestamps_datetime64_ns_median_t[i-1] + np.timedelta64(pause_seconds, 's')
timestamps_int64_ns_median_t = timestamps_datetime64_ns_median_t.astype(np.int64)

time_windows_config_median_t = {
    "10s": np.timedelta64(10, 's'), "5min": np.timedelta64(5, 'm'),
    "30min": np.timedelta64(30, 'm'), "1h": np.timedelta64(1, 'h'),
    "24h": np.timedelta64(24, 'h')
}
MAX_COUNT_WINDOW_FOR_TIME_TEST_median = N_POINTS_TIME_TEST_median 

warmup_array_median_t = np.random.rand(100).astype(np.float64)
warmup_timestamps_median_t_ns = timestamps_int64_ns_median_t[:100] 
warmup_time_window_median_t_ns = np.timedelta64(1, 's').astype('timedelta64[ns]').astype(np.int64)
# Assuming rolling_median_t takes quantile as its last argument
_ = rolling_median_t(warmup_array_median_t, warmup_timestamps_median_t_ns, warmup_time_window_median_t_ns, 100) 

for pd_win_str_median_t, time_window_td_median_t in time_windows_config_median_t.items():
    time_window_ns_for_numba_median_t = time_window_td_median_t.astype('timedelta64[ns]').astype(np.int64)
    start_time_numba_median_t = time.perf_counter()
    # Pass 0.5 as the quantile for rolling_median_t
    _ = rolling_median_t(random_numpy_array, timestamps_int64_ns_median_t, 
                         time_window_ns_for_numba_median_t, MAX_COUNT_WINDOW_FOR_TIME_TEST_median)
    end_time_numba_median_t = time.perf_counter()
    numba_median_t_time = end_time_numba_median_t - start_time_numba_median_t
    pandas_series_time_indexed_median_t = pd.Series(random_numpy_array, index=pd.to_datetime(timestamps_datetime64_ns_median_t))
    start_time_pandas_median_t = time.perf_counter()
    _ = pandas_series_time_indexed_median_t.rolling(window=pd_win_str_median_t, min_periods=1).median()
    end_time_pandas_median_t = time.perf_counter()
    pandas_median_t_time = end_time_pandas_median_t - start_time_pandas_median_t
    if numba_median_t_time > 0 and pandas_median_t_time > 0:
        speedup_median_t = pandas_median_t_time / numba_median_t_time
    else:
        speedup_median_t = np.nan
    benchmark_results.append({
        "Function": "rolling_median_t (time)",
        "Quantile/Window": f"q=0.5, W={pd_win_str_median_t}", 
        "timerollstat (s)": f"{numba_median_t_time:.4f}",
        "Pandas (s)": f"{pandas_median_t_time:.4f}",
        "Speedup": f"{speedup_median_t:.2f}x" if not np.isnan(speedup_median_t) else "N/A"
    })


# ==============================================================================
# NEWLY ADDED BLOCK FOR TIME-BASED ROLLING_QUANTILE_T (various q)
# ==============================================================================
print("\n\n--- Benchmarking: rolling_quantile_t (time-based window, various q) ---")
print("This section benchmarks rolling quantile calculations based on a fixed time duration window for various quantiles.")

# Re-use or regenerate timestamps if necessary. For consistency, let's use a new set or be explicit.
# Using the same N_POINTS as other time tests for simplicity.
N_POINTS_TIME_TEST_quantile = array_size 
np.random.seed(44) # Different seed for potentially different timestamp pattern, or use seed 43
start_datetime_quantile_t = np.datetime64('2023-02-01T00:00:00') # Different start for clarity
timestamps_datetime64_ns_quantile_t = np.empty(N_POINTS_TIME_TEST_quantile, dtype='datetime64[ns]')
timestamps_datetime64_ns_quantile_t[0] = start_datetime_quantile_t
for i in range(1, N_POINTS_TIME_TEST_quantile):
    pause_seconds = np.random.randint(1, 11) 
    timestamps_datetime64_ns_quantile_t[i] = timestamps_datetime64_ns_quantile_t[i-1] + np.timedelta64(pause_seconds, 's')
timestamps_int64_ns_quantile_t = timestamps_datetime64_ns_quantile_t.astype(np.int64)

# Time windows and quantiles to test for rolling_quantile_t
time_windows_config_quantile_t = { # Can be the same as for median_t
    "10s": np.timedelta64(10, 's'), "5min": np.timedelta64(5, 'm'),
    "30min": np.timedelta64(30, 'm'), "1h": np.timedelta64(1, 'h'),
    "24h": np.timedelta64(24, 'h')
}
quantiles_to_test_time_based = [0.1, 0.25, 0.5, 0.75, 0.9] # Same list as count-based, or different
MAX_COUNT_WINDOW_FOR_TIME_TEST_quantile = N_POINTS_TIME_TEST_quantile 

# Warm-up Numba function (rolling_quantile_t)
warmup_array_quantile_t = np.random.rand(100).astype(np.float64)
warmup_timestamps_quantile_t_ns = timestamps_int64_ns_quantile_t[:100] 
warmup_time_window_quantile_t_ns = np.timedelta64(1, 's').astype('timedelta64[ns]').astype(np.int64)
_ = rolling_quantile_t(warmup_array_quantile_t, warmup_timestamps_quantile_t_ns, 
                       warmup_time_window_quantile_t_ns, 100, 0.5) # Warm up with q=0.5
# print("Numba rolling_quantile_t warmed up.")


for pd_win_str_qt, time_window_td_qt in time_windows_config_quantile_t.items():
    time_window_ns_for_numba_qt = time_window_td_qt.astype('timedelta64[ns]').astype(np.int64)
    
    for q_val_time in quantiles_to_test_time_based:
        # print(f"\n-- Testing Time Window: {pd_win_str_qt}, Quantile: {q_val_time} --")

        # --- Numba (Your rolling_quantile_t function) - Time Measurement ---
        start_time_numba_qt = time.perf_counter()
        _ = rolling_quantile_t(random_numpy_array, 
                               timestamps_int64_ns_quantile_t, 
                               time_window_ns_for_numba_qt, 
                               MAX_COUNT_WINDOW_FOR_TIME_TEST_quantile,
                               q_val_time) # Pass the current quantile
        end_time_numba_qt = time.perf_counter()
        numba_time_qt = end_time_numba_qt - start_time_numba_qt
        # print(f"Numba rolling_quantile_t (W={pd_win_str_qt}, q={q_val_time}) time: {numba_time_qt:.4f}s")

        # --- Pandas rolling().quantile() with time window - Time Measurement ---
        pandas_series_time_indexed_qt = pd.Series(random_numpy_array, index=pd.to_datetime(timestamps_datetime64_ns_quantile_t))
        
        start_time_pandas_qt = time.perf_counter()
        _ = pandas_series_time_indexed_qt.rolling(window=pd_win_str_qt, min_periods=1).quantile(q_val_time) # Pass q_val_time
        end_time_pandas_qt = time.perf_counter()
        pandas_time_qt = end_time_pandas_qt - start_time_pandas_qt
        # print(f"Pandas rolling().quantile() (W={pd_win_str_qt}, q={q_val_time}) time: {pandas_time_qt:.4f}s")

        if numba_time_qt > 0 and pandas_time_qt > 0:
            speedup_qt = pandas_time_qt / numba_time_qt
        else:
            speedup_qt = np.nan

        benchmark_results.append({
            "Function": "rolling_quantile_t (time)",
            "Quantile/Window": f"q={q_val_time}, W={pd_win_str_qt}", 
            "timerollstat (s)": f"{numba_time_qt:.4f}",
            "Pandas (s)": f"{pandas_time_qt:.4f}",
            "Speedup": f"{speedup_qt:.2f}x" if not np.isnan(speedup_qt) else "N/A"
        })


# ==============================================================================
# GENERATE MARKDOWN TABLE MANUALLY 
# ==============================================================================
print("\n\n--- Benchmark Summary (Markdown Table) ---")
print(f"\nParameters used for all tests: Array size: {array_size:,}")
print(f"(Count-based tests used window_k: {window_k_count_based:,})")
print(f"(Time-based tests used a large count window and variable time windows)") # Updated message

col_func = "Function"
col_qw = "Quantile/Window" 
col_timeroll_time = "timerollstat (s)"
col_pandas_time = f"Pandas {pd.__version__} (s)"
col_speedup = "Speedup (Pandas/timerollstat)"

if benchmark_results:
    w_func = max(len(col_func), max(len(str(r.get("Function", ""))) for r in benchmark_results)) + 2
    w_qw = max(len(col_qw), max(len(str(r.get("Quantile/Window", ""))) for r in benchmark_results)) + 2
    w_timeroll = max(len(col_timeroll_time), max(len(str(r.get("timerollstat (s)", ""))) for r in benchmark_results)) + 2
    w_pandas = max(len(col_pandas_time), max(len(str(r.get("Pandas (s)", ""))) for r in benchmark_results)) + 2
    w_speedup = max(len(col_speedup), max(len(str(r.get("Speedup", ""))) for r in benchmark_results)) + 2
else: 
    w_func, w_qw, w_timeroll, w_pandas, w_speedup = (len(c) + 2 for c in [col_func, col_qw, col_timeroll_time, col_pandas_time, col_speedup])

header_str = f"| {col_func:<{w_func-2}} | {col_qw:<{w_qw-2}} | {col_timeroll_time:>{w_timeroll-2}} | {col_pandas_time:>{w_pandas-2}} | {col_speedup:>{w_speedup-2}} |"
separator_str = f"|{'-' * w_func}|{'-' * w_qw}|{'-' * w_timeroll}|{'-' * w_pandas}|{'-' * w_speedup}|"

print(header_str)
print(separator_str)

for result in benchmark_results:
    row_str = (
        f"| {str(result.get('Function', 'N/A')):<{w_func-2}} "
        f"| {str(result.get('Quantile/Window', 'N/A')):<{w_qw-2}} " 
        f"| {str(result.get('timerollstat (s)', 'N/A')):>{w_timeroll-2}} "
        f"| {str(result.get('Pandas (s)', 'N/A')):>{w_pandas-2}} "
        f"| {str(result.get('Speedup', 'N/A')):>{w_speedup-2}} |"
    )
    print(row_str)

print("\nCopy the table above and paste it into your README.md")