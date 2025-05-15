import math
import numpy as np
from numba import int64, float64, njit, types, void, optional
import numpy.typing as npt
from typing import Tuple, TypeAlias

# --- Type Aliases and Constants (assuming they are defined as before) ---
PyStateBundle2Type: TypeAlias = Tuple[
    npt.NDArray[np.int64],      # cyclic_window_timestamps
    npt.NDArray[np.float64],    # cyclic_window_values
    npt.NDArray[np.int64],      # _state_arr (head, tail, fill_size, k_window_size, time_window)
    npt.NDArray[np.float64]     # accumulator_arr (single float64 element array for sum)
]

NB_VOID = void
NB_INT64 = int64
NB_FLOAT64 = float64
NB_FLOAT64_ARRAY = float64[:]
NB_INT64_ARRAY = int64[:]

NB_INT64 = types.int64
NB_FLOAT64 = types.float64
PY_INT = int
PY_FLOAT = float


IDX_AVG_HEAD = 0
IDX_AVG_TAIL = 1
IDX_AVG_FILL_SIZE = 2
IDX_AVG_K_WINDOW_SIZE = 3
IDX_AVG_TIMEWINDOW_WINDOW = 4
INT64MAX = 9223372036854775807

STATE_BUNDLE2_TYPE = types.Tuple((
    int64[:],                   # cyclic_window_timestamps
    float64[:],                 # cyclic_window_values
    int64[:],                   # _state_arr
    float64[:]                  # accumulator_arr
))

@njit(STATE_BUNDLE2_TYPE(NB_INT64, NB_INT64), cache=True)
def _init_rolling2_numba(window_size: PY_INT, 
                         window_time: PY_INT
                         ) -> PyStateBundle2Type:
    _k = np.int64(window_size)

    cyclic_window_timestamps_local = np.zeros(_k, dtype=np.int64)
    cyclic_window_values_local = np.zeros(_k, dtype=np.float64)
    
    _state_arr_local = np.zeros(5, dtype=np.int64) 
    _state_arr_local[IDX_AVG_HEAD] = 0 
    _state_arr_local[IDX_AVG_TAIL] = 0 
    _state_arr_local[IDX_AVG_FILL_SIZE] = 0 
    _state_arr_local[IDX_AVG_K_WINDOW_SIZE] = _k 
    _state_arr_local[IDX_AVG_TIMEWINDOW_WINDOW] = window_time 

    accumulator_arr_local = np.zeros(1, dtype=np.float64)

    return (cyclic_window_timestamps_local, 
            cyclic_window_values_local,
            _state_arr_local,
            accumulator_arr_local)

def init_rolling2(window_size: int = 1000, 
                  window_time: int = INT64MAX
                 ) -> PyStateBundle2Type:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_time <= 0 : 
        raise ValueError("window_time must be > 0")
        
    ws_nb = np.int64(window_size)
    wt_nb = np.int64(window_time)
    return _init_rolling2_numba(ws_nb, wt_nb)

@njit(NB_FLOAT64(STATE_BUNDLE2_TYPE, NB_FLOAT64, NB_INT64), fastmath=True, boundscheck=False, cache=True)
def get_rolling_average_t(
    state_tuple2: PyStateBundle2Type,
    new_value: PY_FLOAT,
    new_time: PY_INT
) -> PY_FLOAT:
    cyclic_window_timestamps, \
    cyclic_window_values, \
    _state_arr, \
    accumulator_arr = state_tuple2

    # Copy accumulator value to a local scalar variable for operations
    current_accum_scalar = accumulator_arr[0]

    k_window_size = _state_arr[IDX_AVG_K_WINDOW_SIZE]
    time_window_limit = _state_arr[IDX_AVG_TIMEWINDOW_WINDOW]
    
    # Use local variables for head and fill_size for clarity and potentially minor optimization,
    # then update _state_arr at the end of relevant sections.
    current_head = _state_arr[IDX_AVG_HEAD]
    current_fill_size = _state_arr[IDX_AVG_FILL_SIZE]
    
    # Remove elements expired by time
    while current_fill_size > 0 and \
          (new_time - cyclic_window_timestamps[current_head] > time_window_limit):
        value_to_remove = cyclic_window_values[current_head]
        current_accum_scalar -= value_to_remove # Operate on local scalar
        current_head = (current_head + 1) % k_window_size
        current_fill_size -= 1
        
    # Update state in _state_arr after the time-based removal loop
    _state_arr[IDX_AVG_HEAD] = current_head
    # _state_arr[IDX_AVG_FILL_SIZE] will be updated after adding the new element

    # Remove an element if the window is full by count (after time-based removals)
    if current_fill_size == k_window_size:
        value_to_remove = cyclic_window_values[_state_arr[IDX_AVG_HEAD]] # Use potentially updated head
        current_accum_scalar -= value_to_remove # Operate on local scalar
        _state_arr[IDX_AVG_HEAD] = (_state_arr[IDX_AVG_HEAD] + 1) % k_window_size
        current_fill_size -=1 
    
    # Add the new element
    current_tail = _state_arr[IDX_AVG_TAIL]
    cyclic_window_values[current_tail] = new_value
    cyclic_window_timestamps[current_tail] = new_time
    current_accum_scalar += new_value # Operate on local scalar
    
    _state_arr[IDX_AVG_TAIL] = (current_tail + 1) % k_window_size
    if current_fill_size < k_window_size: # Increment fill_size if it was not full
        current_fill_size += 1
    
    # Final update of state in _state_arr and accumulator_arr
    _state_arr[IDX_AVG_FILL_SIZE] = current_fill_size
    accumulator_arr[0] = current_accum_scalar # Write the final sum back to the state array

    if current_fill_size == 0:
        return np.float64(np.nan)
    
    return current_accum_scalar / current_fill_size