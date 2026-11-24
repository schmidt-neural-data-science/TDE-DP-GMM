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


def _run_ends(states, batch_size):
    for start in range(1, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        changes = states[start:stop] != states[start - 1:stop - 1]
        yield from start + np.flatnonzero(changes)
    if len(states):
        yield len(states)


def _window_mode(window, missing_value):
    labels, counts = np.unique(window[window != missing_value], return_counts=True)
    return labels[np.argmax(counts)]


def imputing_mode_batched(
    states, *, min_samples, window_size=None, missing_value=-1, batch_size=4096
):
    """Impute one continuous 1-D sequence in time batches of up to batch_size samples.

    Runs and voting windows can span batches. Filled samples vote on later
    samples, with ties resolved in favor of the smallest label. If all runs
    are masked, return the original labels.
    """
    if window_size is None:
        window_size = min_samples

    original = np.asarray(states, dtype=int)
    if original.ndim != 1:
        raise ValueError("states must be a one-dimensional sequence")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    x = original.copy()
    run_start = 0
    for end in _run_ends(original, batch_size):
        if end - run_start < min_samples:
            x[run_start:end] = missing_value
        run_start = end

    batches = range(0, len(x), batch_size)
    radius = max(1, int(window_size // 2))

    if x.size and x[0] == missing_value:
        for start in batches:
            known = x[start:start + batch_size] != missing_value
            if known.any():
                first_known = start + int(known.argmax())
                break
        else:
            return original

        # Match the first expanded window that reaches a surviving label.
        first_radius = ((first_known + radius - 1) // radius) * radius
        x[0] = _window_mode(x[first_known:first_radius + 1], missing_value)

    for start in batches:
        missing = np.flatnonzero(x[start:start + batch_size] == missing_value)
        for i in start + missing:
            # The preceding sample is now filled, so the base window has a voter.
            window = x[max(0, i - radius):i + radius + 1]
            x[i] = _window_mode(window, missing_value)
    return x
