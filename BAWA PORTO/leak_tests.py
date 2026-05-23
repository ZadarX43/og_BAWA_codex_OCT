# leak_tests.py
import sys
import numpy as np
from sklearn.metrics import accuracy_score
import os

def assert_no_shuffle_leak(y_true, preds, *, thresh: float = 0.45):
    """\
    Check for potential label leakage by measuring accuracy after shuffling labels.

    If the accuracy of predictions against shuffled labels exceeds ``thresh``, this
    is a red flag for possible leakage.

    Parameters
    ----------
    y_true : array-like
        The *original* target labels.
    preds : array-like
        Model predictions on the same design matrix (probabilities or class
        labels). If a 2D array of shape (n_samples, 3) is passed, the function
        will take ``argmax`` over axis 1 to obtain class labels.
    thresh : float, default 0.45
        Maximum tolerated accuracy after shuffling. ~0.33 is chance level in a
        3-class FTR setting; 0.45 gives a buffer above random before we treat
        it as suspicious. You can override this per-call or via upstream logic.

    Behaviour
    ---------
    - If STRICT_SHUFFLE_LEAK=1 (default), an AssertionError is raised when
      the shuffled-label accuracy is >= ``thresh``.
    - If STRICT_SHUFFLE_LEAK=0, the function only logs a warning and returns
      without raising, so training can continue while still surfacing the
      potential issue in logs.
    """
    # If the caller passed probabilities, take argmax to get labels
    if hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] == 3:
        preds = preds.argmax(axis=1)

    # Defensive conversion to numpy arrays
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    # Shuffle labels deterministically for reproducibility
    rng = np.random.RandomState(1)
    y_shuffled = rng.permutation(y_true)
    acc = accuracy_score(y_shuffled, preds)

    strict = os.getenv("STRICT_SHUFFLE_LEAK", "1").lower() in {"1", "true", "yes", "y"}

    msg = (
        f"❌  Shuffle-label accuracy {acc:.3f} ≥ {thresh:.2f} "
        f"→ leakage suspected for shuffled-label check."
    )

    if acc >= thresh:
        # In strict mode, fail fast; in soft mode, just warn.
        if strict:
            print(msg, file=sys.stderr, flush=True)
            raise AssertionError(msg)
        else:
            try:
                print(msg + " (STRICT_SHUFFLE_LEAK=0 – continuing)", file=sys.stderr, flush=True)
            except Exception:
                pass
    else:
        # Helpful positive signal when the guard passes
        try:
            print(
                f"✅  Shuffle-label accuracy {acc:.3f} < {thresh:.2f} – no obvious leak.",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass