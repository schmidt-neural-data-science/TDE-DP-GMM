import numpy as np

def safe_corrcoef(x, y):
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return np.corrcoef(x, y)[0, 1]




def find_spurious_states(
        *,
        states,
        min_samples,
        target_state = "all", #or state id
):


    states = np.asarray(states, dtype=int)

    replacement = int(states.max()) + 1
    constrained = states.copy()


    run_start = 0
    for idx in range(1, len(states) + 1):
        if idx == len(states) or states[idx] != states[run_start]:
            run_state = states[run_start]
            run_len   = idx - run_start

            # decide if this run should be inspected
            inspect = (
                (target_state == "all" and run_state != replacement) or
                (target_state != "all" and run_state == target_state)
            )

            if inspect and run_len < min_samples:
                constrained[run_start:idx] = replacement

            run_start = idx

    return constrained



def _imputing_mode(x, *, missing_value, window_size):
    """
    imputing most frequent state into missing states (ie, spurious states) by considering a certain window size around the missing value
    :param x:
    :param missing_value:
    :param window_size:
    :return:
    """
    n = x.size
    missing_ids = np.flatnonzero(x == missing_value)
    missing_mask = x == missing_value
    if missing_mask.all():
        print("no missing values")
        return x
    x_copy = x.copy()
    base_radius = max(1, int(window_size//2))

    for i in missing_ids:
        radius = base_radius

        #initial window
        L = max(0, i - radius)
        R = min(n, i + radius + 1)
        win = x_copy[L:R]
        win = win[win != missing_value]

        while win.size == 0:

            radius += base_radius
            L = max(0, i - radius)
            R = min(n, i + radius + 1)
            win = x_copy[L:R]
            win = win[win != missing_value]

            if L == 0 and R == n:  # nowhere left to expand
                break


        vals, counts = np.unique(win, return_counts=True)
        x[i] = vals[np.argmax(counts)]


    return np.asarray(x, dtype=int)




def imputing_mode(states, *, min_samples, window_size = None):

    if window_size is None:
        window_size = min_samples

    states = np.asarray(states, dtype=int)


    states = find_spurious_states(states=states,
                                      min_samples=min_samples,)

    missing_value = int(states.max())  # spurious states are stored as a new state

    states = _imputing_mode(states, missing_value=missing_value, window_size= window_size)
    return np.asarray(states, dtype = int)






