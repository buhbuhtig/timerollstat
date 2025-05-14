import math
import numpy as np
from numba import int64, float64, njit, types, void, optional# Numba типы, используемые в сигнатурах @njit
import numpy.typing as npt
from typing import Tuple, TypeAlias

# Python-совместимый тип для state_bundle
PyStateBundleType: TypeAlias = Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64]
]

# =============================================================================
# Constants for _state array indices (остаются как есть)
# =============================================================================
IDX_STATE_HEAD = 0
IDX_STATE_TAIL = 1
IDX_STATE_FILL_SIZE = 2
IDX_STATE_K_WINDOW_SIZE = 3
IDX_STATE_TGT_QUANTILE_SCALED = 4
IDX_STATE_TIMEWINDOW_WINDOW = 5
INT64MAX = 9223372036854775807

# =============================================================================
# Numba Type for state_bundle (остается как есть, используется в @njit)
# =============================================================================
STATE_BUNDLE_TYPE = types.Tuple((
    float64[:], int64[:], float64[:], int64[:], int64[:], int64[:], int64[:]
))

# =============================================================================
# Helper Numba-типы для читаемости сигнатур @njit (остаются как есть)
# =============================================================================
NB_VOID = void
NB_INT64 = int64
NB_FLOAT64 = float64
NB_FLOAT64_ARRAY = float64[:]
NB_INT64_ARRAY = int64[:]

# =============================================================================
# Python-совместимые типы для аннотаций (новые псевдонимы для удобства)
# =============================================================================
PY_INT = int
PY_FLOAT = float
PY_FLOAT_ARRAY = npt.NDArray[np.float64]
PY_INT_ARRAY = npt.NDArray[np.int64]

# =============================================================================
# Existing Standalone @njit helper functions
# =============================================================================

@njit(NB_VOID(NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _sift_up_max_heap(current_heap_idx: PY_INT,
                      max_heap_values: PY_FLOAT_ARRAY,
                      max_heap_idxs: PY_INT_ARRAY,
                      cyclic_window_tracker: PY_INT_ARRAY) -> None:
    parent_heap_idx = current_heap_idx >> 1
    value_to_sift = max_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = max_heap_idxs[current_heap_idx]
    while current_heap_idx > 1 and value_to_sift > max_heap_values[parent_heap_idx]:
        max_heap_values[current_heap_idx] = max_heap_values[parent_heap_idx]
        max_heap_idxs[current_heap_idx] = max_heap_idxs[parent_heap_idx]
        cyclic_window_tracker[max_heap_idxs[current_heap_idx]] = current_heap_idx
        current_heap_idx = parent_heap_idx
        parent_heap_idx = current_heap_idx >> 1
    max_heap_values[current_heap_idx] = value_to_sift
    max_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    cyclic_window_tracker[tracker_pos_of_sifted_item] = current_heap_idx

@njit(NB_VOID(NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _sift_up_min_heap(current_heap_idx: PY_INT,
                      min_heap_values: PY_FLOAT_ARRAY,
                      min_heap_idxs: PY_INT_ARRAY,
                      cyclic_window_tracker: PY_INT_ARRAY) -> None:
    parent_heap_idx = current_heap_idx >> 1
    value_to_sift = min_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = min_heap_idxs[current_heap_idx]
    while current_heap_idx > 1 and value_to_sift < min_heap_values[parent_heap_idx]:
        min_heap_values[current_heap_idx] = min_heap_values[parent_heap_idx]
        min_heap_idxs[current_heap_idx] = min_heap_idxs[parent_heap_idx]
        cyclic_window_tracker[min_heap_idxs[current_heap_idx]] = -current_heap_idx
        current_heap_idx = parent_heap_idx
        parent_heap_idx = current_heap_idx >> 1
    min_heap_values[current_heap_idx] = value_to_sift
    min_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    cyclic_window_tracker[tracker_pos_of_sifted_item] = -current_heap_idx

@njit(NB_VOID(NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _sift_down_max_heap(current_heap_idx: PY_INT,
                        max_heap_values: PY_FLOAT_ARRAY,
                        max_heap_idxs: PY_INT_ARRAY,
                        cyclic_window_tracker: PY_INT_ARRAY) -> None:
    _heap_sz = max_heap_idxs[0]
    value_to_sift = max_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = max_heap_idxs[current_heap_idx]
    while True:
        child_idx = current_heap_idx << 1
        if child_idx > _heap_sz: break
        right_child_idx = child_idx + 1
        if right_child_idx <= _heap_sz and max_heap_values[right_child_idx] > max_heap_values[child_idx]:
            child_idx = right_child_idx
        if value_to_sift >= max_heap_values[child_idx]: break
        max_heap_values[current_heap_idx] = max_heap_values[child_idx]
        max_heap_idxs[current_heap_idx] = max_heap_idxs[child_idx]
        cyclic_window_tracker[max_heap_idxs[current_heap_idx]] = current_heap_idx
        current_heap_idx = child_idx
    max_heap_values[current_heap_idx] = value_to_sift
    max_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    cyclic_window_tracker[tracker_pos_of_sifted_item] = current_heap_idx

@njit(NB_VOID(NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _sift_down_min_heap(current_heap_idx: PY_INT,
                        min_heap_values: PY_FLOAT_ARRAY,
                        min_heap_idxs: PY_INT_ARRAY,
                        cyclic_window_tracker: PY_INT_ARRAY) -> None:
    _heap_sz = min_heap_idxs[0]
    value_to_sift = min_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = min_heap_idxs[current_heap_idx]
    while True:
        child_idx = current_heap_idx << 1
        if child_idx > _heap_sz: break
        right_child_idx = child_idx + 1
        if right_child_idx <= _heap_sz and min_heap_values[right_child_idx] < min_heap_values[child_idx]:
            child_idx = right_child_idx
        if value_to_sift <= min_heap_values[child_idx]: break
        min_heap_values[current_heap_idx] = min_heap_values[child_idx]
        min_heap_idxs[current_heap_idx] = min_heap_idxs[child_idx]
        cyclic_window_tracker[min_heap_idxs[current_heap_idx]] = -current_heap_idx
        current_heap_idx = child_idx
    min_heap_values[current_heap_idx] = value_to_sift
    min_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    cyclic_window_tracker[tracker_pos_of_sifted_item] = -current_heap_idx

@njit(NB_VOID(NB_FLOAT64, NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _add_element_to_max_heap(value_to_add: PY_FLOAT, # np.float64 in your original code, PY_FLOAT is fine for Python hint
                             window_tracker_slot_for_new_element: PY_INT,
                             max_heap_values: PY_FLOAT_ARRAY,
                             max_heap_idxs: PY_INT_ARRAY,
                             cyclic_window_tracker: PY_INT_ARRAY) -> None:
    max_heap_idxs[0] += 1
    _new_element_heap_idx = max_heap_idxs[0]
    max_heap_values[_new_element_heap_idx] = value_to_add
    max_heap_idxs[_new_element_heap_idx] = window_tracker_slot_for_new_element
    cyclic_window_tracker[window_tracker_slot_for_new_element] = _new_element_heap_idx
    _sift_up_max_heap(_new_element_heap_idx, max_heap_values, max_heap_idxs, cyclic_window_tracker)

@njit(NB_VOID(NB_FLOAT64, NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _add_element_to_min_heap(value_to_add: PY_FLOAT, # np.float64 in your original code, PY_FLOAT is fine for Python hint
                             window_tracker_slot_for_new_element: PY_INT,
                             min_heap_values: PY_FLOAT_ARRAY,
                             min_heap_idxs: PY_INT_ARRAY,
                             cyclic_window_tracker: PY_INT_ARRAY) -> None:
    min_heap_idxs[0] += 1
    _new_element_heap_idx = min_heap_idxs[0]
    min_heap_values[_new_element_heap_idx] = value_to_add
    min_heap_idxs[_new_element_heap_idx] = window_tracker_slot_for_new_element
    cyclic_window_tracker[window_tracker_slot_for_new_element] = -_new_element_heap_idx
    _sift_up_min_heap(_new_element_heap_idx, min_heap_values, min_heap_idxs, cyclic_window_tracker)

@njit(NB_VOID(NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _remove_element_from_max_heap(heap_idx_to_remove: PY_INT,
                                  max_heap_values: PY_FLOAT_ARRAY,
                                  max_heap_idxs: PY_INT_ARRAY,
                                  cyclic_window_tracker: PY_INT_ARRAY) -> None:
    _sift_up_needed = False
    _current_actual_heap_size = max_heap_idxs[0]
    if heap_idx_to_remove < 1 or heap_idx_to_remove > _current_actual_heap_size: return
    if heap_idx_to_remove == _current_actual_heap_size:
        max_heap_idxs[0] -= 1
        return
    _last_element_value = max_heap_values[_current_actual_heap_size]
    _last_element_tracker_idx = max_heap_idxs[_current_actual_heap_size]
    max_heap_values[heap_idx_to_remove] = _last_element_value
    max_heap_idxs[heap_idx_to_remove] = _last_element_tracker_idx
    cyclic_window_tracker[_last_element_tracker_idx] = heap_idx_to_remove
    max_heap_idxs[0] -= 1
    _new_heap_size_after_removal = max_heap_idxs[0]
    if _new_heap_size_after_removal > 0 and heap_idx_to_remove <= _new_heap_size_after_removal:
        _parent_idx = heap_idx_to_remove // 2
        if heap_idx_to_remove > 1 and max_heap_values[heap_idx_to_remove] > max_heap_values[_parent_idx]:
            _sift_up_needed = True
        if _sift_up_needed:
            _sift_up_max_heap(heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        else:
            _sift_down_max_heap(heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)

@njit(NB_VOID(NB_INT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _remove_element_from_min_heap(heap_idx_to_remove: PY_INT,
                                  min_heap_values: PY_FLOAT_ARRAY,
                                  min_heap_idxs: PY_INT_ARRAY,
                                  cyclic_window_tracker: PY_INT_ARRAY) -> None:
    _sift_up_needed = False
    _current_actual_heap_size = min_heap_idxs[0]
    if heap_idx_to_remove < 1 or heap_idx_to_remove > _current_actual_heap_size: return
    if heap_idx_to_remove == _current_actual_heap_size:
        min_heap_idxs[0] -= 1
        return
    _last_element_value = min_heap_values[_current_actual_heap_size]
    _last_element_tracker_idx = min_heap_idxs[_current_actual_heap_size]
    min_heap_values[heap_idx_to_remove] = _last_element_value
    min_heap_idxs[heap_idx_to_remove] = _last_element_tracker_idx
    cyclic_window_tracker[_last_element_tracker_idx] = -heap_idx_to_remove
    min_heap_idxs[0] -= 1
    _new_heap_size_after_removal = min_heap_idxs[0]
    if _new_heap_size_after_removal > 0 and heap_idx_to_remove <= _new_heap_size_after_removal:
        _parent_idx = heap_idx_to_remove // 2
        if heap_idx_to_remove > 1 and min_heap_values[heap_idx_to_remove] < min_heap_values[_parent_idx]:
           _sift_up_needed = True
        if _sift_up_needed:
            _sift_up_min_heap(heap_idx_to_remove, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        else:
            _sift_down_min_heap(heap_idx_to_remove, min_heap_values, min_heap_idxs, cyclic_window_tracker)

@njit(NB_VOID(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _rebalance_single_step_for_median(
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY,
    cyclic_window_tracker: PY_INT_ARRAY) -> None:
    current_max_heap_size = max_heap_idxs[0]
    current_min_heap_size = min_heap_idxs[0]
    if current_max_heap_size > current_min_heap_size + 1:
        if current_max_heap_size > 0:
            value_to_move = max_heap_values[1]
            tracker_idx_of_value_to_move = max_heap_idxs[1]
            _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)
            _add_element_to_min_heap(value_to_move, tracker_idx_of_value_to_move, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    elif current_min_heap_size > current_max_heap_size:
        if current_min_heap_size > 0:
            value_to_move = min_heap_values[1]
            tracker_idx_of_value_to_move = min_heap_idxs[1]
            _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
            _add_element_to_max_heap(value_to_move, tracker_idx_of_value_to_move, max_heap_values, max_heap_idxs, cyclic_window_tracker)

@njit(NB_VOID(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY,NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _rebalance_single_step_for_quantile(
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY,
    cyclic_window_tracker: PY_INT_ARRAY,
    _state: PY_INT_ARRAY) -> None:
    
    N = _state[IDX_STATE_FILL_SIZE]
    tgt_quantile = _state[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0
    if N == 1:
        target_size_max_heap = np.int64(1)
    else:
        k_rank_0_based = tgt_quantile * (N - 1.0)
        target_size_max_heap = np.int64(math.floor(k_rank_0_based) + 1)

    if target_size_max_heap < 0:
        target_size_max_heap = 0
    if target_size_max_heap > N:
        target_size_max_heap = N
    current_max_heap_size = max_heap_idxs[0]
    current_min_heap_size = min_heap_idxs[0]

    if current_max_heap_size > target_size_max_heap:
        if current_max_heap_size > 0:
            value_to_move = max_heap_values[1]
            tracker_idx_of_value_to_move = max_heap_idxs[1]
            _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)
            _add_element_to_min_heap(value_to_move, tracker_idx_of_value_to_move, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    elif current_max_heap_size < target_size_max_heap:
        if current_min_heap_size > 0:
            value_to_move = min_heap_values[1]
            tracker_idx_of_value_to_move = min_heap_idxs[1]
            _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
            _add_element_to_max_heap(value_to_move, tracker_idx_of_value_to_move, max_heap_values, max_heap_idxs, cyclic_window_tracker)

@njit(NB_VOID(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _rebalance_multi_step_for_median( # Renamed for brevity
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY,
    cyclic_window_tracker: PY_INT_ARRAY) -> None:

    # Loop to balance heap sizes
    while not (max_heap_idxs[0] == min_heap_idxs[0] or max_heap_idxs[0] == min_heap_idxs[0] + 1):
        if max_heap_idxs[0] > min_heap_idxs[0] + 1: # Max-heap is too large
            # Move an element from max-heap to min-heap
            value_to_move = max_heap_values[1]
            tracker_idx_of_value_to_move = max_heap_idxs[1]
            _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)
            _add_element_to_min_heap(value_to_move, tracker_idx_of_value_to_move, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        elif min_heap_idxs[0] > max_heap_idxs[0]: # Min-heap is too large
            # Move an element from min-heap to max-heap
            value_to_move = min_heap_values[1]
            tracker_idx_of_value_to_move = min_heap_idxs[1]
            _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
            _add_element_to_max_heap(value_to_move, tracker_idx_of_value_to_move, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        # else: # Sizes are already balanced according to the while loop's condition
            # break # This break is not needed as the while condition will terminate the loop.
            # If we reach here, the while condition should be false, and the loop will end.
            # If the while condition is still true, but neither if/elif triggered,
            # it implies a logical error in the while or if/elif conditions.
            # However, for median calculation, these conditions cover all imbalance cases.

@njit(NB_VOID(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY),
      fastmath=True, boundscheck=False, cache=True)
def _rebalance_multi_step_for_quantile(
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY,
    cyclic_window_tracker: PY_INT_ARRAY,
    _state: PY_INT_ARRAY) -> None:

    N = _state[IDX_STATE_FILL_SIZE]
    tgt_quantile = _state[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0

    if N == 1:
        target_size_max_heap = np.int64(1)
    else:
        k_rank_0_based = tgt_quantile * (N - 1.0)
        target_size_max_heap = np.int64(math.floor(k_rank_0_based) + 1)

    if target_size_max_heap < 0:
        target_size_max_heap = 0
    if target_size_max_heap > N:
        target_size_max_heap = N

    # Loop to balance heap sizes toward target
    while max_heap_idxs[0] != target_size_max_heap:
        if max_heap_idxs[0] > target_size_max_heap:
            if max_heap_idxs[0] > 0:
                value_to_move = max_heap_values[1]
                tracker_idx_of_value_to_move = max_heap_idxs[1]
                _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)
                _add_element_to_min_heap(value_to_move, tracker_idx_of_value_to_move, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        elif max_heap_idxs[0] < target_size_max_heap:
            if min_heap_idxs[0] > 0:
                value_to_move = min_heap_values[1]
                tracker_idx_of_value_to_move = min_heap_idxs[1]
                _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
                _add_element_to_max_heap(value_to_move, tracker_idx_of_value_to_move, max_heap_values, max_heap_idxs, cyclic_window_tracker)



@njit(NB_FLOAT64(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _get_median_from_balanced_heaps(
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY
) -> PY_FLOAT:
    max_h_size = max_heap_idxs[0]
    min_h_size = min_heap_idxs[0]
    if max_h_size == 0 and min_h_size == 0: return np.float64(np.nan)
    if max_h_size > min_h_size: return max_heap_values[1]
    elif min_h_size > max_h_size: return min_heap_values[1]
    else: return (max_heap_values[1] + min_heap_values[1]) / 2.0

'''
@njit(NB_VOID(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _rebalance_single_step_for_quantile(
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY,
    cyclic_window_tracker: PY_INT_ARRAY, # Numba signature has int64[:]
    _state: PY_INT_ARRAY # Numba signature has int64[:]
) -> None:
    N = _state[IDX_STATE_FILL_SIZE]
    tgt_quantile = _state[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0
    if N == 0: return
    if N == 1:
        target_size_max_heap = np.int64(1)
    else:
        k_rank_0_based = tgt_quantile * (N - 1.0)
        idx_low_0_based = math.floor(k_rank_0_based)
        target_size_max_heap = np.int64(idx_low_0_based + 1)
    if target_size_max_heap < 0: target_size_max_heap = 0
    if target_size_max_heap > N: target_size_max_heap = N
    current_max_heap_size = max_heap_idxs[0]
    if current_max_heap_size > target_size_max_heap:
        if current_max_heap_size > 0:
            value_to_move = max_heap_values[1]
            tracker_idx_of_value_to_move = max_heap_idxs[1]
            _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)
            _add_element_to_min_heap(value_to_move, tracker_idx_of_value_to_move, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    elif current_max_heap_size < target_size_max_heap:
        if min_heap_idxs[0] > 0:
            value_to_move = min_heap_values[1]
            tracker_idx_of_value_to_move = min_heap_idxs[1]
            _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
            _add_element_to_max_heap(value_to_move, tracker_idx_of_value_to_move, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        current_max_heap_size_after_size_rebalance = max_heap_idxs[0]
    
    current_max_heap_size_after_size_rebalance = max_heap_idxs[0]
    current_min_heap_size_after_size_rebalance = min_heap_idxs[0]

    if current_max_heap_size_after_size_rebalance > 0 and \
       current_min_heap_size_after_size_rebalance > 0 and \
       max_heap_values[1] > min_heap_values[1]:
        
        val_from_max = max_heap_values[1]
        idx_tracker_from_max = max_heap_idxs[1]

        val_from_min = min_heap_values[1]
        idx_tracker_from_min = min_heap_idxs[1]

        _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)

        _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
        
        
        _add_element_to_max_heap(val_from_min, idx_tracker_from_min, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        
        _add_element_to_min_heap(val_from_max, idx_tracker_from_max, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    
'''

@njit(NB_FLOAT64(NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64, NB_FLOAT64), fastmath=True, boundscheck=False, cache = True)
def _get_quantile_from_balanced_heaps(
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY,
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY,
    current_tracker_fill_size: PY_INT,
    tgt_quantile: PY_FLOAT
) -> PY_FLOAT:
    N = current_tracker_fill_size
    max_h_size = max_heap_idxs[0]
    min_h_size = min_heap_idxs[0]
    quantile_value: float64 # type: ignore
    if N == 1:
        if max_h_size == 1:
            return max_heap_values[1]
        else:
            return np.float64(np.nan)
    if tgt_quantile == 0.5:
        if max_h_size > min_h_size:
            quantile_value = max_heap_values[1]
        elif min_h_size > max_h_size:
            quantile_value = min_heap_values[1]
        else:
            quantile_value = (max_heap_values[1] + min_heap_values[1]) / 2.0
        return quantile_value
    k_rank_0_based = tgt_quantile * (N - 1.0)
    idx_low_0_based = math.floor(k_rank_0_based)
    val_low_valid: bool = (max_h_size > 0)
    val_high_valid: bool = (min_h_size > 0)
    val_low: float64 = np.float64(0.0) # type: ignore
    if val_low_valid:
        val_low = max_heap_values[1]
    val_high: float64 = np.float64(0.0) # type: ignore
    if val_high_valid:
        val_high = min_heap_values[1]
    if idx_low_0_based == k_rank_0_based:
        if val_low_valid:
            quantile_value = val_low
        elif val_high_valid:
            quantile_value = val_high
        else:
            quantile_value = np.float64(np.nan)
    else:
        if val_low_valid and val_high_valid:
            h = k_rank_0_based - idx_low_0_based
            quantile_value = val_low * (1.0 - h) + val_high * h
        elif val_low_valid:
            quantile_value = val_low
        elif val_high_valid:
            quantile_value = val_high
        else:
            quantile_value = np.float64(np.nan)
    return quantile_value

@njit(NB_VOID(NB_FLOAT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _perform_update_tracker_and_add_for_median(
    new_value: PY_FLOAT, # Numba signature NB_FLOAT64
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY, # Numba signature NB_INT64_ARRAY
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY, # Numba signature NB_INT64_ARRAY
    cyclic_window_tracker: PY_INT_ARRAY, # Numba signature NB_INT64_ARRAY
    _state: PY_INT_ARRAY # Numba signature NB_INT64_ARRAY
) -> None:
    k_window_size = _state[IDX_STATE_K_WINDOW_SIZE]
    if _state[IDX_STATE_FILL_SIZE] == k_window_size:
        _oldest_element_tracker_slot = _state[IDX_STATE_HEAD]
        _signed_heap_idx_to_remove: int64 = cyclic_window_tracker[_oldest_element_tracker_slot] # type: ignore
        if _signed_heap_idx_to_remove > 0:
            _remove_element_from_max_heap(_signed_heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        elif _signed_heap_idx_to_remove < 0:
            _remove_element_from_min_heap(abs(_signed_heap_idx_to_remove), min_heap_values, min_heap_idxs, cyclic_window_tracker)
        _state[IDX_STATE_HEAD] = (_state[IDX_STATE_HEAD] + 1) % k_window_size
    else:
        _state[IDX_STATE_FILL_SIZE] += 1
    _new_element_tracker_slot = _state[IDX_STATE_TAIL]
    _current_max_heap_size = max_heap_idxs[0]
    if _current_max_heap_size == 0 or (_current_max_heap_size > 0 and new_value <= max_heap_values[1]):
        if min_heap_idxs[0] > 0 and new_value > min_heap_values[1]: # to keep invariant rule
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        else:#base scenario (no invariant violation, so add to the heap that supports a balanced state)
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
    else:
        if max_heap_idxs[0] > 0 and new_value < max_heap_values[1]:# to keep invariant rule
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        else: #base scenario (no invariant violation, so add to the heap that supports a balanced state)
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    
    _state[IDX_STATE_TAIL] = (_state[IDX_STATE_TAIL] + 1) % k_window_size

@njit(NB_VOID(NB_FLOAT64, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_FLOAT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY, NB_INT64_ARRAY), fastmath=True, boundscheck=False, cache = True)
def _perform_update_tracker_and_add_for_quantile(
    new_value: PY_FLOAT, # Numba signature NB_FLOAT64
    max_heap_values: PY_FLOAT_ARRAY,
    max_heap_idxs: PY_INT_ARRAY, # Numba signature NB_INT64_ARRAY
    min_heap_values: PY_FLOAT_ARRAY,
    min_heap_idxs: PY_INT_ARRAY, # Numba signature NB_INT64_ARRAY
    cyclic_window_tracker: PY_INT_ARRAY, # Numba signature NB_INT64_ARRAY
    _state: PY_INT_ARRAY # Numba signature NB_INT64_ARRAY
) -> None:
    k_window_size = _state[IDX_STATE_K_WINDOW_SIZE]
    if _state[IDX_STATE_FILL_SIZE] == k_window_size:
        _oldest_element_tracker_slot = _state[IDX_STATE_HEAD]
        _signed_heap_idx_to_remove: int64 = cyclic_window_tracker[_oldest_element_tracker_slot] # type: ignore
        if _signed_heap_idx_to_remove > 0:
            _remove_element_from_max_heap(_signed_heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        elif _signed_heap_idx_to_remove < 0:
            _remove_element_from_min_heap(abs(_signed_heap_idx_to_remove), min_heap_values, min_heap_idxs, cyclic_window_tracker)
        _state[IDX_STATE_HEAD] = (_state[IDX_STATE_HEAD] + 1) % k_window_size
    else:
        _state[IDX_STATE_FILL_SIZE] += 1
    _new_element_tracker_slot = _state[IDX_STATE_TAIL]
    _current_max_heap_size = max_heap_idxs[0]
    N = _state[IDX_STATE_FILL_SIZE]
    tgt_quantile = _state[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0
    k_rank_0_based = tgt_quantile * (N - 1.0)
    target_size_max_heap = np.int64(math.floor(k_rank_0_based) + 1)

    if _current_max_heap_size < target_size_max_heap:
        if min_heap_idxs[0] > 0 and new_value > min_heap_values[1]: # to keep invariant rule
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        else:#base scenario (no invariant violation, so add to the heap that supports a balanced state)
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
    else:
        if max_heap_idxs[0] > 0 and new_value < max_heap_values[1]:# to keep invariant rule
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        else: #base scenario (no invariant violation, so add to the heap that supports a balanced state)
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    
    _state[IDX_STATE_TAIL] = (_state[IDX_STATE_TAIL] + 1) % k_window_size


@njit(NB_VOID(STATE_BUNDLE_TYPE, NB_INT64), fastmath=True, boundscheck=False, cache=True)
def _remove_expired_by_time_and_size(state_tuple: PyStateBundleType, new_time: PY_INT) -> None:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, cyclic_window_timestamps, _state_arr = state_tuple
    k_window_size = _state_arr[IDX_STATE_K_WINDOW_SIZE]
    time_window_limit = _state_arr[IDX_STATE_TIMEWINDOW_WINDOW]
    while _state_arr[IDX_STATE_FILL_SIZE] > 0 and \
          (new_time - cyclic_window_timestamps[_state_arr[IDX_STATE_HEAD]] > time_window_limit):
        _oldest_element_tracker_slot = _state_arr[IDX_STATE_HEAD]
        _signed_heap_idx_to_remove = cyclic_window_tracker[_oldest_element_tracker_slot]
        if _signed_heap_idx_to_remove > 0:
            _remove_element_from_max_heap(_signed_heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        elif _signed_heap_idx_to_remove < 0:
            _remove_element_from_min_heap(abs(_signed_heap_idx_to_remove), min_heap_values, min_heap_idxs, cyclic_window_tracker)
        _state_arr[IDX_STATE_HEAD] = (_state_arr[IDX_STATE_HEAD] + 1) % k_window_size
        _state_arr[IDX_STATE_FILL_SIZE] -= 1
    if _state_arr[IDX_STATE_FILL_SIZE] == k_window_size:
        _oldest_element_tracker_slot = _state_arr[IDX_STATE_HEAD]
        _signed_heap_idx_to_remove = cyclic_window_tracker[_oldest_element_tracker_slot]
        if _signed_heap_idx_to_remove > 0:
            _remove_element_from_max_heap(_signed_heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        elif _signed_heap_idx_to_remove < 0:
            _remove_element_from_min_heap(abs(_signed_heap_idx_to_remove), min_heap_values, min_heap_idxs, cyclic_window_tracker)
        _state_arr[IDX_STATE_HEAD] = (_state_arr[IDX_STATE_HEAD] + 1) % k_window_size
        _state_arr[IDX_STATE_FILL_SIZE] -= 1

@njit(NB_VOID(STATE_BUNDLE_TYPE, NB_FLOAT64, NB_INT64), fastmath=True, boundscheck=False, cache=True)
def _add_new_element_with_time_median(state_tuple: PyStateBundleType, new_value: PY_FLOAT, new_time: PY_INT) -> None:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, cyclic_window_timestamps, _state_arr = state_tuple
    k_window_size = _state_arr[IDX_STATE_K_WINDOW_SIZE]
    _new_element_tracker_slot = _state_arr[IDX_STATE_TAIL]


    _current_max_heap_size = max_heap_idxs[0]
    if _current_max_heap_size == 0 or (_current_max_heap_size > 0 and new_value <= max_heap_values[1]):
        if min_heap_idxs[0] > 0 and new_value > min_heap_values[1]: # to keep invariant rule
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        else:#base scenario (no invariant violation, so add to the heap that supports a balanced state)
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
    else:
        if max_heap_idxs[0] > 0 and new_value < max_heap_values[1]:# to keep invariant rule
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        else: #base scenario (no invariant violation, so add to the heap that supports a balanced state)
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)

    cyclic_window_timestamps[_new_element_tracker_slot] = new_time
    _state_arr[IDX_STATE_TAIL] = (_state_arr[IDX_STATE_TAIL] + 1) % k_window_size
    if _state_arr[IDX_STATE_FILL_SIZE] < k_window_size:
        _state_arr[IDX_STATE_FILL_SIZE] += 1

@njit(NB_VOID(STATE_BUNDLE_TYPE, NB_FLOAT64, NB_INT64), fastmath=True, boundscheck=False, cache=True)
def _add_new_element_with_time_quantile(state_tuple: PyStateBundleType, new_value: PY_FLOAT, new_time: PY_INT) -> None:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, cyclic_window_timestamps, _state_arr = state_tuple

    k_window_size = _state_arr[IDX_STATE_K_WINDOW_SIZE]
    _new_element_tracker_slot = _state_arr[IDX_STATE_TAIL]

    _current_max_heap_size = max_heap_idxs[0]
    N = _state_arr[IDX_STATE_FILL_SIZE] + 1  # we're adding a new element
    tgt_quantile = _state_arr[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0
    k_rank_0_based = tgt_quantile * (N - 1.0)
    target_size_max_heap = np.int64(math.floor(k_rank_0_based) + 1)

    if _current_max_heap_size < target_size_max_heap:
        if min_heap_idxs[0] > 0 and new_value > min_heap_values[1]:
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
        else:
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
    else:
        if max_heap_idxs[0] > 0 and new_value < max_heap_values[1]:
            _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        else:
            _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)

    cyclic_window_timestamps[_new_element_tracker_slot] = new_time
    _state_arr[IDX_STATE_TAIL] = (_state_arr[IDX_STATE_TAIL] + 1) % k_window_size
    if _state_arr[IDX_STATE_FILL_SIZE] < k_window_size:
        _state_arr[IDX_STATE_FILL_SIZE] += 1



def init_rolling(window_size: int = 1000,
                 window_time: int = INT64MAX,
                 q: float = 0.5
                ) -> PyStateBundleType:
    ws_nb = np.int64(window_size)
    wt_nb = np.int64(window_time)
    q_nb = np.float64(q)
    return _init_rolling_numba(ws_nb, wt_nb, q_nb)


@njit(STATE_BUNDLE_TYPE(NB_INT64, NB_INT64, NB_FLOAT64), cache=True)
def _init_rolling_numba(window_size: PY_INT,
                 window_time: PY_INT,
                 quantile: PY_FLOAT,
) -> PyStateBundleType: # Python аннотация использует PyStateBundleType
    if window_size < 1:
        raise ValueError('Should be: window_size >= 1')
    if window_time <= 0:
        raise ValueError('Should be: time_window > 0')
    _k = np.int64(window_size)
    _heap_capacity = _k + 1
    max_heap_values_local = np.empty(_heap_capacity, dtype=np.float64)
    max_heap_idxs_local = np.zeros(_heap_capacity, dtype=np.int64)
    min_heap_values_local = np.empty(_heap_capacity, dtype=np.float64)
    min_heap_idxs_local = np.zeros(_heap_capacity, dtype=np.int64)
    cyclic_window_tracker_local = np.empty(_k, dtype=np.int64)
    cyclic_window_timestamps_local = np.zeros(_k, dtype=np.int64)
    _state_arr_local = np.zeros(6, dtype=np.int64)
    _state_arr_local[IDX_STATE_K_WINDOW_SIZE] = _k
    _tgt_q_internal = quantile
    if _tgt_q_internal < 0.0: _tgt_q_internal = 0.0
    elif _tgt_q_internal > 1.0: _tgt_q_internal = 1.0
    _state_arr_local[IDX_STATE_TGT_QUANTILE_SCALED] = np.int64(round(_tgt_q_internal * 10000.0))
    _state_arr_local[IDX_STATE_HEAD] = 0
    _state_arr_local[IDX_STATE_TAIL] = 0
    _state_arr_local[IDX_STATE_FILL_SIZE] = 0
    _state_arr_local[IDX_STATE_TIMEWINDOW_WINDOW] = window_time
    return (max_heap_values_local, max_heap_idxs_local,
            min_heap_values_local, min_heap_idxs_local,
            cyclic_window_tracker_local, cyclic_window_timestamps_local,
            _state_arr_local)

@njit(NB_FLOAT64(STATE_BUNDLE_TYPE, NB_FLOAT64), fastmath=True, boundscheck=False, cache=True)
def get_median(
    state_tuple: PyStateBundleType,
    new_value: PY_FLOAT
) -> PY_FLOAT:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple
    _perform_update_tracker_and_add_for_median(
        new_value,
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        cyclic_window_tracker, _state_arr
    )
    if _state_arr[IDX_STATE_FILL_SIZE] == 0:
        return np.float64(np.nan)
    _rebalance_single_step_for_median(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        cyclic_window_tracker
    )
    return _get_median_from_balanced_heaps(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs
    )

@njit(NB_FLOAT64(STATE_BUNDLE_TYPE, NB_FLOAT64, NB_INT64), fastmath=True, boundscheck=False, cache=True)
def get_median_t(
    state_tuple: PyStateBundleType,
    new_value: PY_FLOAT,
    new_time: PY_INT
) -> PY_FLOAT:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple
    _remove_expired_by_time_and_size(state_tuple, new_time)
    _add_new_element_with_time_median(state_tuple, new_value, new_time)
    if _state_arr[IDX_STATE_FILL_SIZE] == 0:
        return np.float64(np.nan)
    _rebalance_multi_step_for_median(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        cyclic_window_tracker
    )
    return _get_median_from_balanced_heaps(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs
    )

@njit(NB_FLOAT64(STATE_BUNDLE_TYPE, NB_FLOAT64, NB_INT64), fastmath=True, boundscheck=False, cache=True)
def get_quantile_t(
    state_tuple: PyStateBundleType,
    new_value: PY_FLOAT,
    new_time: PY_INT
) -> PY_FLOAT:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple
    _remove_expired_by_time_and_size(state_tuple, new_time)
    _add_new_element_with_time_quantile(state_tuple, new_value, new_time)
    if _state_arr[IDX_STATE_FILL_SIZE] == 0:
        return np.float64(np.nan)
    _rebalance_multi_step_for_quantile(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        cyclic_window_tracker,
        _state_arr
    )
    return _get_quantile_from_balanced_heaps(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        _state_arr[IDX_STATE_FILL_SIZE],
        _state_arr[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0
    )

@njit(NB_FLOAT64(STATE_BUNDLE_TYPE, NB_FLOAT64), fastmath=True, boundscheck=False, cache=True)
def get_quantile(
    state_tuple: PyStateBundleType,
    new_value: PY_FLOAT
) -> PY_FLOAT:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple
    _perform_update_tracker_and_add_for_quantile(
        new_value,
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        cyclic_window_tracker, _state_arr
    )
    if _state_arr[IDX_STATE_FILL_SIZE] == 0:
        return np.float64(np.nan)
    _rebalance_single_step_for_quantile(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        cyclic_window_tracker, _state_arr
    )
    return _get_quantile_from_balanced_heaps(
        max_heap_values, max_heap_idxs,
        min_heap_values, min_heap_idxs,
        _state_arr[IDX_STATE_FILL_SIZE],
        _state_arr[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0
    )

@njit(NB_FLOAT64_ARRAY(NB_FLOAT64_ARRAY, NB_INT64), fastmath=True, boundscheck=False, cache = True)
def rolling_median(input_array: PY_FLOAT_ARRAY, # Было float64[:]
                           window_size: PY_INT) -> PY_FLOAT_ARRAY: # Было float64[:]
    n = len(input_array)
    output_medians = np.empty(n, dtype=np.float64)
    if window_size < 1:
        raise ValueError('Should be: k_window_size_param >= 1')
    if n == 0:
        return output_medians
    state_tuple = _init_rolling_numba(window_size, np.iinfo(np.int64).max, 0.5)
    for i in range(n):
        output_medians[i] = get_median(state_tuple, input_array[i])
    return output_medians

@njit(NB_FLOAT64_ARRAY(NB_FLOAT64_ARRAY, NB_INT64_ARRAY,NB_INT64,NB_INT64,NB_FLOAT64), fastmath=True, boundscheck=False, cache = True)
def rolling_quantile_t(input_array: PY_FLOAT_ARRAY,
                     datatimes: PY_INT_ARRAY,
                     window_time:PY_INT_ARRAY,
                    window_size: PY_INT,
                    quantile: PY_FLOAT) -> PY_FLOAT_ARRAY:
    n = len(input_array)
    output_quantiles = np.empty(n, dtype=np.float64)
    if window_size < 1:
        raise ValueError('Should be: k_window_size_param >= 1')
    if n == 0:
        return output_quantiles
    state_tuple = _init_rolling_numba(window_size, window_time, quantile)
    for i in range(n):
        output_quantiles[i] = get_quantile_t(state_tuple, input_array[i], datatimes[i])
    return output_quantiles

@njit(NB_FLOAT64_ARRAY(NB_FLOAT64_ARRAY, NB_INT64_ARRAY,NB_INT64,NB_INT64), fastmath=True, boundscheck=False, cache = True)
def rolling_median_t(input_array: PY_FLOAT_ARRAY,
                     datatimes: PY_INT_ARRAY,
                     window_time:PY_INT_ARRAY,
                    window_size: PY_INT) -> PY_FLOAT_ARRAY: # Было float64[:]
    n = len(input_array)
    output_medians = np.empty(n, dtype=np.float64)
    if window_size < 1:
        raise ValueError('Should be: k_window_size_param >= 1')
    if n == 0:
        return output_medians
    state_tuple = _init_rolling_numba(window_size, window_time, 0.5)
    for i in range(n):
        output_medians[i] = get_median_t(state_tuple, input_array[i], datatimes[i])
    return output_medians

@njit(NB_FLOAT64_ARRAY(NB_FLOAT64_ARRAY, NB_INT64, NB_FLOAT64), fastmath=True, boundscheck=False, cache = True)
def rolling_quantile(input_array: PY_FLOAT_ARRAY, # Было float64[:]
                             window_size: PY_INT, # Было int64
                             quantile: PY_FLOAT) -> PY_FLOAT_ARRAY: # Было float64[:]
    n = len(input_array)
    output_quantiles = np.empty(n, dtype=np.float64)
    if window_size < 1:
        raise ValueError('Should be: k_window_size_param >= 1')
    if n == 0:
        return output_quantiles
    if not (0.0 <= quantile <= 1.0):
        raise ValueError('tgt_quantile_param must be between 0.0 and 1.0')
    state_tuple = _init_rolling_numba(window_size, np.iinfo(np.int64).max, quantile)
    # _state_arr_ref = state_tuple[6] # This variable was unused
    for i in range(n):
        output_quantiles[i] = get_quantile(state_tuple, input_array[i])
    return output_quantiles