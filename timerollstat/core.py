import math
import numpy as np
from numba import int64, float64, njit, types, void

# =============================================================================
# Constants for _state array indices
# =============================================================================
IDX_STATE_HEAD = 0
IDX_STATE_TAIL = 1
IDX_STATE_FILL_SIZE = 2
IDX_STATE_K_WINDOW_SIZE = 3
IDX_STATE_TGT_QUANTILE_SCALED = 4
IDX_STATE_TIMEWINDOW_WINDOW = 5 # New index for time window

# =============================================================================
# Type for state_bundle (tuple of array types) - now 7 elements
# (max_vals, max_idxs, min_vals, min_idxs, tracker, timestamps, _state_arr)
# =============================================================================
STATE_BUNDLE_TYPE = types.Tuple((
    float64[:], int64[:], float64[:], int64[:], int64[:], int64[:], int64[:]
))

# =============================================================================
# Existing Standalone @njit helper functions (NO CHANGES)
# _sift_*, _add_element_to_*_heap, _remove_element_from_*_heap,
# _rebalance_single_step_for_median, _get_median_from_balanced_heaps (old version),
# _rebalance_single_step_for_quantile, _get_quantile_from_balanced_heaps (old version),
# _perform_update_tracker_and_add (old version)
# =============================================================================

@njit(void(int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _sift_up_max_heap(current_heap_idx: int64,
                      max_heap_values: float64[:],
                      max_heap_idxs: int64[:], # max_heap_idxs[heap_pos] = tracker_pos
                      cyclic_window_tracker: int64[:]): # cyclic_window_tracker[tracker_pos] = heap_pos
    parent_heap_idx = current_heap_idx >> 1
    
    # Store the value and tracker_idx of the element that "sifts up"
    value_to_sift = max_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = max_heap_idxs[current_heap_idx]

    while current_heap_idx > 1 and value_to_sift > max_heap_values[parent_heap_idx]:
        # Move the parent element down
        max_heap_values[current_heap_idx] = max_heap_values[parent_heap_idx]
        max_heap_idxs[current_heap_idx] = max_heap_idxs[parent_heap_idx]
        
        # Update the tracker for the parent element moved down
        cyclic_window_tracker[max_heap_idxs[current_heap_idx]] = current_heap_idx
        
        current_heap_idx = parent_heap_idx
        parent_heap_idx = current_heap_idx >> 1
    
    # Place the "sifted up" element in its final position
    max_heap_values[current_heap_idx] = value_to_sift
    max_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    
    # Update the tracker for the "sifted up" element once
    cyclic_window_tracker[tracker_pos_of_sifted_item] = current_heap_idx

@njit(void(int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _sift_up_min_heap(current_heap_idx: int64,
                      min_heap_values: float64[:],
                      min_heap_idxs: int64[:],
                      cyclic_window_tracker: int64[:]):
    parent_heap_idx = current_heap_idx >> 1

    value_to_sift = min_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = min_heap_idxs[current_heap_idx]

    while current_heap_idx > 1 and value_to_sift < min_heap_values[parent_heap_idx]:
        min_heap_values[current_heap_idx] = min_heap_values[parent_heap_idx]
        min_heap_idxs[current_heap_idx] = min_heap_idxs[parent_heap_idx]
        cyclic_window_tracker[min_heap_idxs[current_heap_idx]] = -current_heap_idx # Minus sign for min_heap
        
        current_heap_idx = parent_heap_idx
        parent_heap_idx = current_heap_idx >> 1
        
    min_heap_values[current_heap_idx] = value_to_sift
    min_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    cyclic_window_tracker[tracker_pos_of_sifted_item] = -current_heap_idx # Minus sign for min_heap

@njit(void(int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _sift_down_max_heap(current_heap_idx: int64, # This is the index of the element to be sifted down
                        max_heap_values: float64[:],
                        max_heap_idxs: int64[:],
                        cyclic_window_tracker: int64[:]):
    _heap_sz = max_heap_idxs[0]
    
    # Store the value and tracker_idx of the element that "sinks down"
    value_to_sift = max_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = max_heap_idxs[current_heap_idx]

    while True:
        child_idx = current_heap_idx << 1 # Start with the left child
        if child_idx > _heap_sz: # No children
            break
        
        # Choose the larger of the two children
        right_child_idx = child_idx + 1
        if right_child_idx <= _heap_sz and max_heap_values[right_child_idx] > max_heap_values[child_idx]:
            child_idx = right_child_idx
            
        # If the "sinking" element is larger than its largest child, it's in place
        if value_to_sift >= max_heap_values[child_idx]: # Use >= for stability if values are equal
            break
            
        # Move the larger child up
        max_heap_values[current_heap_idx] = max_heap_values[child_idx]
        max_heap_idxs[current_heap_idx] = max_heap_idxs[child_idx]
        
        # Update the tracker for the child moved up
        cyclic_window_tracker[max_heap_idxs[current_heap_idx]] = current_heap_idx
        
        current_heap_idx = child_idx # Move to the next level down
        
    # Place the "sunk" element in its final position
    max_heap_values[current_heap_idx] = value_to_sift
    max_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    
    # Update the tracker for the "sunk" element once
    cyclic_window_tracker[tracker_pos_of_sifted_item] = current_heap_idx

@njit(void(int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _sift_down_min_heap(current_heap_idx: int64,
                        min_heap_values: float64[:],
                        min_heap_idxs: int64[:],
                        cyclic_window_tracker: int64[:]):
    _heap_sz = min_heap_idxs[0]

    value_to_sift = min_heap_values[current_heap_idx]
    tracker_pos_of_sifted_item = min_heap_idxs[current_heap_idx]

    while True:
        child_idx = current_heap_idx << 1
        if child_idx > _heap_sz:
            break
        
        right_child_idx = child_idx + 1
        if right_child_idx <= _heap_sz and min_heap_values[right_child_idx] < min_heap_values[child_idx]:
            child_idx = right_child_idx
            
        if value_to_sift <= min_heap_values[child_idx]: # Use <= for stability
            break
            
        min_heap_values[current_heap_idx] = min_heap_values[child_idx]
        min_heap_idxs[current_heap_idx] = min_heap_idxs[child_idx]
        cyclic_window_tracker[min_heap_idxs[current_heap_idx]] = -current_heap_idx # Minus sign
        
        current_heap_idx = child_idx
        
    min_heap_values[current_heap_idx] = value_to_sift
    min_heap_idxs[current_heap_idx] = tracker_pos_of_sifted_item
    cyclic_window_tracker[tracker_pos_of_sifted_item] = -current_heap_idx # Minus sign

@njit(void(float64, int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _add_element_to_max_heap(value_to_add: np.float64,
                             window_tracker_slot_for_new_element: int64,
                             max_heap_values: float64[:],
                             max_heap_idxs: int64[:],
                             cyclic_window_tracker: int64[:]):
    max_heap_idxs[0] += 1
    _new_element_heap_idx = max_heap_idxs[0]
    max_heap_values[_new_element_heap_idx] = value_to_add
    max_heap_idxs[_new_element_heap_idx] = window_tracker_slot_for_new_element
    cyclic_window_tracker[window_tracker_slot_for_new_element] = _new_element_heap_idx
    _sift_up_max_heap(_new_element_heap_idx, max_heap_values, max_heap_idxs, cyclic_window_tracker)

@njit(void(float64, int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _add_element_to_min_heap(value_to_add: np.float64,
                             window_tracker_slot_for_new_element: int64,
                             min_heap_values: float64[:],
                             min_heap_idxs: int64[:],
                             cyclic_window_tracker: int64[:]):
    min_heap_idxs[0] += 1
    _new_element_heap_idx = min_heap_idxs[0]
    min_heap_values[_new_element_heap_idx] = value_to_add
    min_heap_idxs[_new_element_heap_idx] = window_tracker_slot_for_new_element
    cyclic_window_tracker[window_tracker_slot_for_new_element] = -_new_element_heap_idx
    _sift_up_min_heap(_new_element_heap_idx, min_heap_values, min_heap_idxs, cyclic_window_tracker)

@njit(void(int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _remove_element_from_max_heap(heap_idx_to_remove: int64,
                                  max_heap_values: float64[:],
                                  max_heap_idxs: int64[:],
                                  cyclic_window_tracker: int64[:]):
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

@njit(void(int64, float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _remove_element_from_min_heap(heap_idx_to_remove: int64,
                                  min_heap_values: float64[:],
                                  min_heap_idxs: int64[:],
                                  cyclic_window_tracker: int64[:]):
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

@njit(void(float64[:], int64[:], float64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _rebalance_single_step_for_median(
    max_heap_values: float64[:],
    max_heap_idxs: int64[:],
    min_heap_values: float64[:],
    min_heap_idxs: int64[:],
    cyclic_window_tracker: int64[:]):
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

@njit(float64(float64[:], int64[:], float64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _get_median_from_balanced_heaps(
    max_heap_values: float64[:],
    max_heap_idxs: int64[:],
    min_heap_values: float64[:],
    min_heap_idxs: int64[:]
) -> float64:
    max_h_size = max_heap_idxs[0]
    min_h_size = min_heap_idxs[0]
    if max_h_size == 0 and min_h_size == 0: return np.float64(np.nan)
    if max_h_size > min_h_size: return max_heap_values[1]
    elif min_h_size > max_h_size: return min_heap_values[1]
    else: return (max_heap_values[1] + min_heap_values[1]) / 2.0

@njit(void(float64[:], int64[:], float64[:], int64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _rebalance_single_step_for_quantile( # Signature remains the same
    max_heap_values: float64[:],
    max_heap_idxs: int64[:],
    min_heap_values: float64[:],
    min_heap_idxs: int64[:],
    cyclic_window_tracker: int64[:],
    _state: int64[:] # Contains N and tgt_quantile
):
    N = _state[IDX_STATE_FILL_SIZE]
    tgt_quantile = _state[IDX_STATE_TGT_QUANTILE_SCALED] / 10000.0

    if N == 0: return

    # Target size for max_heap so its root is x_i (element at position floor((N-1)*q) )
    if N == 1: # Special case, k_rank will be 0, idx_low will be 0. max_heap must contain 1 element.
        target_size_max_heap = np.int64(1)
    else:
        k_rank_0_based = tgt_quantile * (N - 1.0)
        idx_low_0_based = math.floor(k_rank_0_based)
        target_size_max_heap = np.int64(idx_low_0_based + 1)
    
    # Constraints on target_size_max_heap
    if target_size_max_heap < 0: target_size_max_heap = 0 # Should not happen for N>=1
    if target_size_max_heap > N: target_size_max_heap = N 


    current_max_heap_size = max_heap_idxs[0]
    # Transfer so that max_heap_size becomes equal to target_size_max_heap
    if current_max_heap_size > target_size_max_heap:
        if current_max_heap_size > 0: # Only if there is something to transfer
            value_to_move = max_heap_values[1]
            tracker_idx_of_value_to_move = max_heap_idxs[1]
            _remove_element_from_max_heap(np.int64(1), max_heap_values, max_heap_idxs, cyclic_window_tracker)
            _add_element_to_min_heap(value_to_move, tracker_idx_of_value_to_move, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    elif current_max_heap_size < target_size_max_heap:
        if min_heap_idxs[0] > 0: # Only if there is somewhere to transfer from
            value_to_move = min_heap_values[1]
            tracker_idx_of_value_to_move = min_heap_idxs[1]
            _remove_element_from_min_heap(np.int64(1), min_heap_values, min_heap_idxs, cyclic_window_tracker)
            _add_element_to_max_heap(value_to_move, tracker_idx_of_value_to_move, max_heap_values, max_heap_idxs, cyclic_window_tracker)

@njit(float64(float64[:], int64[:], float64[:], int64[:], int64, float64), fastmath=True, boundscheck=False, cache = True)
def _get_quantile_from_balanced_heaps(
    max_heap_values: float64[:],
    max_heap_idxs: int64[:],
    min_heap_values: float64[:],
    min_heap_idxs: int64[:],
    current_tracker_fill_size: int64,
    tgt_quantile: float64
) -> float64:
    N = current_tracker_fill_size
    # N > 0 is assumed by the calling function (_add_value / _add_value_quantile,
    # which check fill_size before calling this function)
    
    max_h_size = max_heap_idxs[0]
    min_h_size = min_heap_idxs[0]
    quantile_value: float64 

    if N == 1:
        # For N=1, after balancing in _rebalance_single_step_for_quantile,
        # target_size_max_heap will be 1. The element will be in max_heap.
        if max_h_size == 1:
            return max_heap_values[1]
        else:
            # This is an anomaly if N=1 and max_h_size is not 1 after balancing.
            # Return NaN to indicate a problem if this occurs.
            return np.float64(np.nan) 

    if tgt_quantile == 0.5: # Median
        if max_h_size > min_h_size: 
            quantile_value = max_heap_values[1]
        elif min_h_size > max_h_size: 
            quantile_value = min_heap_values[1]
        else: # max_h_size == min_h_size (and both > 0, since N > 1)
            quantile_value = (max_heap_values[1] + min_heap_values[1]) / 2.0
        return quantile_value

    # For other quantiles, including q=0 and q=1 (N > 1)
    k_rank_0_based = tgt_quantile * (N - 1.0)
    idx_low_0_based = math.floor(k_rank_0_based)
    
    val_low_valid: bool = (max_h_size > 0)
    val_high_valid: bool = (min_h_size > 0)

    # val_low and val_high will only be read if the corresponding flag is True
    val_low: float64 = np.float64(0.0) # Initialization for Numba
    if val_low_valid:
        val_low = max_heap_values[1]
    
    val_high: float64 = np.float64(0.0) # Initialization for Numba
    if val_high_valid:
        val_high = min_heap_values[1]

    if idx_low_0_based == k_rank_0_based: # Exact hit on the rank
        if val_low_valid:
            quantile_value = val_low
        # If val_low is invalid (max_h_size=0), this means idx_low_0_based=0,
        # and all elements (if N>0) should be in min_heap.
        # This is covered by the tgt_quantile == 0.0 case below if we move it out.
        # Or, if q=0, then target_size_max_heap in _rebalance will be 0 (or 1 if N=1).
        # If target_size_max_heap=0, then max_h_size=0, val_low_valid=False.
        # min_h_size will be N. val_high_valid=True.
        # Then for q=0, k_rank=0, idx_low_rank=0. We'll land here.
        # If val_low is not valid, but val_high is valid, then this is our minimum.
        elif val_high_valid: # Added for the case q=0, N>1
            quantile_value = val_high
        else: # Both heaps are empty, but N>1 - anomaly
            quantile_value = np.float64(np.nan)
    else: # Linear interpolation is needed
        if val_low_valid and val_high_valid:
            h = k_rank_0_based - idx_low_0_based
            quantile_value = val_low * (1.0 - h) + val_high * h
        elif val_low_valid: # min_heap is empty, but interpolation is needed (e.g., q is close to 1)
            quantile_value = val_low
        elif val_high_valid: # max_heap is empty, but interpolation is needed (e.g., q is close to 0)
            quantile_value = val_high
        else: # Both heaps are empty, but N > 1 - anomaly
            quantile_value = np.float64(np.nan)
            
    return quantile_value

@njit(void(float64, float64[:], int64[:], float64[:], int64[:], int64[:], int64[:]), fastmath=True, boundscheck=False, cache = True)
def _perform_update_tracker_and_add(
    new_value: np.float64,
    max_heap_values: float64[:],
    max_heap_idxs: int64[:],
    min_heap_values: float64[:],
    min_heap_idxs: int64[:],
    cyclic_window_tracker: int64[:],
    _state: int64[:]
):
    k_window_size = _state[IDX_STATE_K_WINDOW_SIZE]
    if _state[IDX_STATE_FILL_SIZE] == k_window_size:
        _oldest_element_tracker_slot = _state[IDX_STATE_HEAD]
        _signed_heap_idx_to_remove: int64 = cyclic_window_tracker[_oldest_element_tracker_slot]
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
         _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
    else:
         _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    _state[IDX_STATE_TAIL] = (_state[IDX_STATE_TAIL] + 1) % k_window_size

# =============================================================================
# NEW and MODIFIED functions for procedural approach with time window
# =============================================================================

@njit(void(STATE_BUNDLE_TYPE, int64), fastmath=True, boundscheck=False, cache=True)
def _remove_expired_by_time_and_size(state_tuple: STATE_BUNDLE_TYPE, new_time: int64):
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, cyclic_window_timestamps, _state_arr = state_tuple

    k_window_size = _state_arr[IDX_STATE_K_WINDOW_SIZE]
    time_window_limit = _state_arr[IDX_STATE_TIMEWINDOW_WINDOW]

    # First, remove elements that are outside the time window
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
    
    # Then, if the window is still full by count, remove one more to make space
    if _state_arr[IDX_STATE_FILL_SIZE] == k_window_size:
        _oldest_element_tracker_slot = _state_arr[IDX_STATE_HEAD]
        _signed_heap_idx_to_remove = cyclic_window_tracker[_oldest_element_tracker_slot]
        if _signed_heap_idx_to_remove > 0:
            _remove_element_from_max_heap(_signed_heap_idx_to_remove, max_heap_values, max_heap_idxs, cyclic_window_tracker)
        elif _signed_heap_idx_to_remove < 0:
            _remove_element_from_min_heap(abs(_signed_heap_idx_to_remove), min_heap_values, min_heap_idxs, cyclic_window_tracker)
        _state_arr[IDX_STATE_HEAD] = (_state_arr[IDX_STATE_HEAD] + 1) % k_window_size
        _state_arr[IDX_STATE_FILL_SIZE] -= 1 # Decrement, as space was freed

@njit(void(STATE_BUNDLE_TYPE, float64, int64), fastmath=True, boundscheck=False, cache=True)
def _add_new_element_with_time(state_tuple: STATE_BUNDLE_TYPE, new_value: float64, new_time: int64):
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, cyclic_window_timestamps, _state_arr = state_tuple

    k_window_size = _state_arr[IDX_STATE_K_WINDOW_SIZE]
    
    _new_element_tracker_slot = _state_arr[IDX_STATE_TAIL]
    
    _current_max_heap_size = max_heap_idxs[0]
    if _current_max_heap_size == 0 or (_current_max_heap_size > 0 and new_value <= max_heap_values[1]):
         _add_element_to_max_heap(new_value, _new_element_tracker_slot, max_heap_values, max_heap_idxs, cyclic_window_tracker)
    else:
         _add_element_to_min_heap(new_value, _new_element_tracker_slot, min_heap_values, min_heap_idxs, cyclic_window_tracker)
    
    cyclic_window_timestamps[_new_element_tracker_slot] = new_time
    _state_arr[IDX_STATE_TAIL] = (_state_arr[IDX_STATE_TAIL] + 1) % k_window_size
    
    if _state_arr[IDX_STATE_FILL_SIZE] < k_window_size:
        _state_arr[IDX_STATE_FILL_SIZE] += 1

@njit(types.Tuple((float64[:], int64[:], float64[:], int64[:], int64[:], int64[:], int64[:]))(int64, int64, float64), cache=True)
def init_rolling(window_size: int64 =100002,
                 time_window:int64 = np.iinfo(np.int64).max, # Use the maximum int64 value as "infinite" time
                 quantile: float64 = 0.5,
) -> STATE_BUNDLE_TYPE:
    if window_size < 1:
        raise ValueError('Should be: window_size >= 1')
    if time_window <= 0:
        raise ValueError('Should be: time_window > 0')
    
    _k = np.int64(window_size)
    _heap_capacity = _k + 1
    max_heap_values_local = np.empty(_heap_capacity, dtype=np.float64)
    max_heap_idxs_local = np.zeros(_heap_capacity, dtype=np.int64)
    min_heap_values_local = np.empty(_heap_capacity, dtype=np.float64)
    min_heap_idxs_local = np.zeros(_heap_capacity, dtype=np.int64)
    cyclic_window_tracker_local = np.empty(_k, dtype=np.int64)
    cyclic_window_timestamps_local = np.zeros(_k, dtype=np.int64) # Initialize with zeros
    
    _state_arr_local = np.zeros(6, dtype=np.int64) # Size 6 for the new IDX_STATE_TIMEWINDOW_WINDOW
    _state_arr_local[IDX_STATE_K_WINDOW_SIZE] = _k
    
    _tgt_q_internal = quantile
    if _tgt_q_internal < 0.0: _tgt_q_internal = 0.0
    elif _tgt_q_internal > 1.0: _tgt_q_internal = 1.0
    _state_arr_local[IDX_STATE_TGT_QUANTILE_SCALED] = np.int64(round(_tgt_q_internal * 10000.0))
    
    _state_arr_local[IDX_STATE_HEAD] = 0
    _state_arr_local[IDX_STATE_TAIL] = 0
    _state_arr_local[IDX_STATE_FILL_SIZE] = 0
    _state_arr_local[IDX_STATE_TIMEWINDOW_WINDOW] = time_window

    return (max_heap_values_local, max_heap_idxs_local,
            min_heap_values_local, min_heap_idxs_local,
            cyclic_window_tracker_local, cyclic_window_timestamps_local, # Added timestamps
            _state_arr_local)

@njit(float64(STATE_BUNDLE_TYPE, float64), fastmath=True, boundscheck=False, cache=True)
def get_median(
    state_tuple: STATE_BUNDLE_TYPE, 
    new_value: float64
) -> float64:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple # cyclic_window_timestamps is not needed for this version

    _perform_update_tracker_and_add(
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

@njit(float64(STATE_BUNDLE_TYPE, float64, int64), fastmath=True, boundscheck=False, cache=True)
def get_median_t(
    state_tuple: STATE_BUNDLE_TYPE, 
    new_value: float64,
    new_time: int64
) -> float64:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple # cyclic_window_timestamps will be used internally

    _remove_expired_by_time_and_size(state_tuple, new_time)
    _add_new_element_with_time(state_tuple, new_value, new_time)
    
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

@njit(float64(STATE_BUNDLE_TYPE, float64), fastmath=True, boundscheck=False, cache=True)
def get_quantile(
    state_tuple: STATE_BUNDLE_TYPE,
    new_value: float64
) -> float64:
    max_heap_values, max_heap_idxs, \
    min_heap_values, min_heap_idxs, \
    cyclic_window_tracker, _, _state_arr = state_tuple # cyclic_window_timestamps is not needed

    _perform_update_tracker_and_add(
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

@njit(float64[:](float64[:], int64), fastmath=True, boundscheck=False, cache = True)
def rolling_median(input_array: float64[:],
                           k_window_size_param: int64) -> float64[:]:
    n = len(input_array)
    output_medians = np.empty(n, dtype=np.float64)

    if k_window_size_param < 1: # Check and ValueError
        raise ValueError('Should be: k_window_size_param >= 1')
    if n == 0: # If the input array is empty, do nothing
        return output_medians 
    
    state_tuple = init_rolling(k_window_size_param, np.iinfo(np.int64).max, 0.5) # default time_window

    for i in range(n):
        output_medians[i] = get_median(state_tuple, input_array[i]) # Assuming get_median is defined
    return output_medians
    
@njit(float64[:](float64[:], int64, float64), fastmath=True, boundscheck=False, cache = True)
def rolling_quantile(input_array: float64[:],
                             k_window_size_param: int64,
                             tgt_quantile_param: float64) -> float64[:]:
    n = len(input_array)
    output_quantiles = np.empty(n, dtype=np.float64)

    if k_window_size_param < 1: # Check and ValueError
        raise ValueError('Should be: k_window_size_param >= 1')
    if n == 0:
        return output_quantiles
    # Additional check for tgt_quantile_param, if not done in init_rolling
    if not (0.0 <= tgt_quantile_param <= 1.0):
        raise ValueError('tgt_quantile_param must be between 0.0 and 1.0')


    state_tuple = init_rolling(k_window_size_param, np.iinfo(np.int64).max, tgt_quantile_param)
    
    _state_arr_ref = state_tuple[6] # Index of _state_arr in the tuple

    for i in range(n):
        output_quantiles[i] = get_quantile(state_tuple, input_array[i]) # Assuming get_quantile is defined
    return output_quantiles