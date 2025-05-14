# TimeRollStat

**High-performance online rolling statistics (median, quantile) over time-based windows, powered by Numba.**

`timerollstat` provides a fast and efficient way to calculate rolling medians and quantiles on streaming data.

This library is ideal for applications requiring real-time statistical analysis of time-series data, such as financial data streams or sensor readings, particularly where new data points arrive sequentially **with uneven time intervals**. It allows you to maintain up-to-date rolling statistics over a look-back time period.

## Key Features

*   **Time-Based Windows**: Define rolling windows by duration (e.g., "last 5 minutes", "last 24 hours").
*   **Handles Uneven Timestamps**: Naturally processes data points that do not arrive at regular intervals.
*   **Online Processing (For-Loop Friendly)**: Efficiently update statistics as new data points arrive one by one, perfect for iterative processing.
*   **High Performance**: Leverages Numba for C-like speed.
*   **Quantiles & Median**: Calculate arbitrary rolling quantiles and medians using efficient heap-based structures.

## Quick Example

Here's how `timerollstat` processes data sequentially for a time-based rolling median:

```python
import numpy as np
from timerollstat import init_rolling, get_median_t

# 1. Sample data and timestamps
values = np.array([10.0, 12.5, 11.0, 15.0, 13.5, 16.0])
# Timestamps as int64 nanoseconds
timestamps_ns = np.array([
    np.datetime64('2023-01-01T10:00:00').astype(np.int64),
    np.datetime64('2023-01-01T10:00:05').astype(np.int64), # +5s
    np.datetime64('2023-01-01T10:00:15').astype(np.int64), # +10s
    np.datetime64('2023-01-01T10:00:32').astype(np.int64), # +17s
    np.datetime64('2023-01-01T10:00:35').astype(np.int64), # +3s
    np.datetime64('2023-01-01T10:00:40').astype(np.int64)  # +5s
])

# 2. Initialize the rolling state
# Max buffer count, time window duration (nanoseconds), quantile (0.5 for median)
rol = init_rolling(window_size=100, # Max buffer (count)
                   time_window=np.int64(30 * 1e9), # 30 seconds in nanoseconds
                   quantile=0.5) # For median

# 3. Process data points iteratively
for i in range(len(values)):
    current_value = values[i]
    current_timestamp_ns = timestamps_ns[i]
    median = get_median_t(rol, current_value, current_timestamp_ns)
    print(f"{current_value:<11.1f} | {current_timestamp_ns:<20} | {median:.2f}")

# 10.0        | 1672567200000000000  | 10.00
# 12.5        | 1672567205000000000  | 11.25
# 11.0        | 1672567215000000000  | 11.00
# 15.0        | 1672567232000000000  | 13.25  (10.0 is now out of 30s window)
# 13.5        | 1672567235000000000  | 13.50  (12.5 is now out of 30s window)
# 16.0        | 1672567240000000000  | 15.00  (11.0 is now out of 30s window)
```

## Performance

`timerollstat` is designed for speed. Its Numba-compiled core processes data **sequentially** and often surpasses the performance of Pandas' **batch** `rolling().quantile()` operations.

**Benchmark Environment:**
*   Array size for all tests: 1,500,000
*   Pandas version: 2.2.3
*   NumPy version: 2.2.5

---

**Benchmark Summary: Count-Based Windows**
Fixed count window: 5,000.
| Method                                 | Window / Details   |   Time (s) |   Speedup (vs Pandas Batch) |
|:---------------------------------------|:-------------------|-----------:|----------------------------:|
| `timerollstat_median` (for-loop online)  | q=0.5              |     0.8150 |                       3.95x |
| `Pandas` (batch)                       | q=0.5              |     3.2220 |                             |
| `timerollstat_quantile` (for-loop online)| q=0.1              |     0.6813 |                       4.49x |
| `Pandas` (batch)                       | q=0.1              |     3.0581 |                             |
| `timerollstat_quantile` (for-loop online)| q=0.25             |     0.7835 |                       3.96x |
| `Pandas` (batch)                       | q=0.25             |     3.1054 |                             |
| `timerollstat_quantile` (for-loop online)| q=0.5              |     0.8281 |                       3.87x |
| `Pandas` (batch)                       | q=0.5              |     3.2049 |                             |
| `timerollstat_quantile` (for-loop online)| q=0.75             |     0.7688 |                       4.13x |
| `Pandas` (batch)                       | q=0.75             |     3.1716 |                             |
| `timerollstat_quantile` (for-loop online)| q=0.9              |     0.6689 |                       4.64x |
| `Pandas` (batch)                       | q=0.9              |     3.1019 |                             |

---

**Benchmark Summary: Time-Based Rolling Median**
Max count window: 1,500,000.
| Method                                  | Window / Details            |   Time (s) |   Speedup (vs Pandas Batch) |
|:----------------------------------------|:----------------------------|-----------:|----------------------------:|
| `timerollstat_median_t` (for-loop online) | q=0.5, 5min  |     1.7516 |                       1.09x |
| `Pandas` (batch)                        | q=0.5, 5min  |     1.9097 |                             |
| `timerollstat_median_t` (for-loop online) | q=0.5, 30min |     1.8707 |                       1.18x |
| `Pandas` (batch)                        | q=0.5, 30min |     2.2034 |                             |
| `timerollstat_median_t` (for-loop online) | q=0.5, 1h    |     1.8335 |                       1.30x |
| `Pandas` (batch)                        | q=0.5, 1h    |     2.3809 |                             |
| `timerollstat_median_t` (for-loop online) | q=0.5, 24h   |     1.9593 |                       2.08x |
| `Pandas` (batch)                        | q=0.5, 24h   |     4.0695 |                             |

---

**Benchmark Summary: Time-Based Rolling Quantile (q=0.25)**
Max count window: 1,500,000.
| Method                                    | Window / Details   |   Time (s) |   Speedup (vs Pandas Batch) |
|:------------------------------------------|:-------------------|-----------:|----------------------------:|
| `timerollstat_quantile_t` (for-loop online) | 5min        |     1.6915 |                       1.12x |
| `Pandas` (batch)                          | 5min        |     1.8981 |                             |
| `timerollstat_quantile_t` (for-loop online) | 30min       |     1.7381 |                       1.26x |
| `Pandas` (batch)                          | 30min       |     2.1915 |                             |
| `timerollstat_quantile_t` (for-loop online) | 1h          |     1.7865 |                       1.33x |
| `Pandas` (batch)                          | 1h          |     2.3822 |                             |
| `timerollstat_quantile_t` (for-loop online) | 24h         |     1.8440 |                       2.22x |
| `Pandas` (batch)                          | 24h         |     4.0967 |                             |

---

**Benchmark Summary: Time-Based Rolling Quantile (q=0.75)**
Max count window: 1,500,000.
| Method                                    | Window / Details   |   Time (s) |   Speedup (vs Pandas Batch) |
|:------------------------------------------|:-------------------|-----------:|----------------------------:|
| `timerollstat_quantile_t` (for-loop online) | 5min        |     1.7038 |                       1.16x |
| `Pandas` (batch)                          | 5min        |     1.9752 |                             |
| `timerollstat_quantile_t` (for-loop online) | 30min       |     1.7713 |                       1.26x |
| `Pandas` (batch)                          | 30min       |     2.2297 |                             |
| `timerollstat_quantile_t` (for-loop online) | 1h          |     1.8324 |                       1.34x |
| `Pandas` (batch)                          | 1h          |     2.4611 |                             |
| `timerollstat_quantile_t` (for-loop online) | 24h         |     1.8200 |                       2.27x |
| `Pandas` (batch)                          | 24h         |     4.1310 |                             |


## Considerations & Limitations

*   **Numba JIT Warm-up**: The first time a Numba-compiled function (e.g., after `init_rolling` for a specific state instance) is called with a particular set of argument types, Numba performs Just-In-Time compilation. This initial call will be slower. Subsequent calls using the same state instance and argument types will benefit from the cached, compiled code. This warm-up is typical for Numba and is excluded from the benchmark figures above, which reflect the performance of already-compiled code.
*   **Memory Usage**: For very large time windows or extremely high data rates, monitor memory usage as the internal data buffers will grow accordingly up to the specified `window_size` (count) or the number of elements fitting the time window.

## Installation

```bash
pip install timerollstat *
```
*Not ready, yet*

Or, for the development version:
```bash
pip install git+https://github.com/buhbuhtig/timerollstat.git
```

## Alternatives

While `timerollstat` is optimized for online, time-based window calculations, other excellent libraries exist for related tasks:

*   **[River](https://riverml.xyz/)**: A comprehensive library for online machine learning in Python. It offers high-performance rolling window statistics based on *count*, but does not natively support time-based duration windows for these specific rolling statistics in the same way `timerollstat` does.
*   **[Bottleneck](https://bottleneck.readthedocs.io/)**: Provides very fast C-optimized NumPy array functions for moving window calculations (e.g., `move_median`, `move_mean`). It is highly performant for *count-based* rolling windows but does not directly support time-based duration windows.

`timerollstat` aims to fill the niche for high-speed, online, **time-duration based** rolling quantiles and medians.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/buhbuhtig/timerollstat).

## License
This project is licensed under the MIT License.
