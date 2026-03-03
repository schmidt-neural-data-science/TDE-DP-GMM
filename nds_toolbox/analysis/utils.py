import numpy as np

def safe_corrcoef(x, y):
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return np.corrcoef(x, y)[0, 1]




def find_spurious_states(*, states, min_samples, target_state="all", missing_value=-1):
    states = np.asarray(states, dtype=int)
    constrained = states.copy()

    run_start = 0
    for idx in range(1, len(states) + 1):
        if idx == len(states) or states[idx] != states[run_start]:
            run_state = states[run_start]
            run_len   = idx - run_start

            inspect = (
                (target_state == "all" and run_state != missing_value) or
                (target_state != "all" and run_state == target_state)
            )

            if inspect and run_len < min_samples:
                constrained[run_start:idx] = missing_value

            run_start = idx

    return constrained



import numpy as np

def _imputing_mode(x, *, missing_value, window_size):
    x = np.asarray(x, dtype=int).copy()
    n = x.size

    missing_ids = np.flatnonzero(x == missing_value)
    if missing_ids.size == 0:
        return x  # no missing values

    if missing_ids.size == n:
        return x  # all missing

    x_copy = x.copy()
    base_radius = max(1, int(window_size // 2))

    for i in missing_ids:
        radius = base_radius
        while True:
            L = max(0, i - radius)
            R = min(n, i + radius + 1)
            win = x_copy[L:R]
            win = win[win != missing_value]

            if win.size > 0:
                vals, counts = np.unique(win, return_counts=True)
                fill = vals[np.argmax(counts)]
                x[i] = fill
                x_copy[i] = fill
                break

            if L == 0 and R == n:
                # nowhere left to expand; leave as missing_value
                break

            radius += base_radius

    return x



def imputing_mode(states, *, min_samples, window_size=None, missing_value=-1):
    if window_size is None:
        window_size = min_samples

    original = np.asarray(states, dtype=int)
    x = original.copy()

    x = find_spurious_states(states=x, min_samples=min_samples, missing_value=missing_value)

    if np.all(x == missing_value):
        return original

    x = _imputing_mode(x, missing_value=missing_value, window_size=window_size)

    if np.all(x == missing_value):
        return original


    return np.asarray(x, dtype=int)





