"""PLR (Pupillary Light Reflex) parameter extraction.

Processes pupil diameter time series to extract clinically relevant
PLR biomarkers via parametric curve fitting.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@dataclass
class PLRParameters:
    """Fitted PLR biomarker parameters."""

    baseline_diameter: float
    constriction_amplitude: float
    constriction_amplitude_pct: float
    constriction_latency: float  # seconds
    peak_constriction_velocity: float  # mm/s (or normalised units/s)
    recovery_time_constant: float  # seconds
    fit_r_squared: float


# ---------------------------------------------------------------------------
# PLR waveform model
# ---------------------------------------------------------------------------


def _plr_model(
    t: np.ndarray, d0: float, A: float, t_onset: float, tau_c: float, tau_d: float
) -> np.ndarray:
    """Parametric PLR waveform.

    d(t) = d0 - A * (1 - exp(-(t - t_onset)/tau_c)) * exp(-(t - t_onset)/tau_d)

    Only active for t > t_onset; returns d0 before onset.
    """
    result = np.full_like(t, d0, dtype=np.float64)
    mask = t > t_onset
    dt = t[mask] - t_onset
    result[mask] = d0 - A * (1.0 - np.exp(-dt / tau_c)) * np.exp(-dt / tau_d)
    return result


def _peak_constriction_velocity(A: float, tau_c: float, tau_d: float) -> float:
    """Analytically derive peak constriction velocity from fitted params.

    The derivative of the constriction term:
      d'(t) = -A * [exp(-dt/tau_c)/tau_c * exp(-dt/tau_d)
                     - (1 - exp(-dt/tau_c)) * exp(-dt/tau_d)/tau_d]

    Peak velocity occurs at t* = tau_c * tau_d / (tau_c - tau_d) * ln(tau_c/tau_d)
    when tau_c != tau_d.
    """
    if tau_c <= 0 or tau_d <= 0:
        return 0.0

    if abs(tau_c - tau_d) < 1e-6:
        # Degenerate case: approximate
        return A / (tau_c * np.e)

    t_peak = tau_c * tau_d / (tau_c - tau_d) * np.log(tau_c / tau_d)
    if t_peak < 0:
        t_peak = 0.0

    # Evaluate |d'(t_peak)|
    e_c = np.exp(-t_peak / tau_c)
    e_d = np.exp(-t_peak / tau_d)
    deriv = A * (e_c / tau_c * e_d - (1.0 - e_c) * e_d / tau_d)
    return abs(deriv)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def smooth_plr(diameters: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing that preserves PLR constriction shape."""
    if len(diameters) < window:
        return diameters.copy()
    return savgol_filter(diameters, window, polyorder)


def detect_blinks(
    diameters: np.ndarray,
    baseline: float,
    threshold_frac: float = 0.5,
    confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """Return boolean mask where True = valid (non-blink) frame."""
    valid = diameters > (baseline * threshold_frac)
    if confidence is not None:
        valid &= confidence > confidence_threshold
    return valid


def reject_outliers(
    diameters: np.ndarray, fps: float, max_velocity_mm_s: float = 8.0
) -> np.ndarray:
    """Reject frames with physiologically impossible diameter changes."""
    max_change = max_velocity_mm_s / fps
    valid = np.ones(len(diameters), dtype=bool)
    for i in range(1, len(diameters)):
        if abs(diameters[i] - diameters[i - 1]) > max_change:
            valid[i] = False
    return valid


def interpolate_invalid(
    diameters: np.ndarray, valid: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    """Linearly interpolate through invalid frames."""
    result = diameters.copy()
    if valid.all() or not valid.any():
        return result
    result[~valid] = np.interp(timestamps[~valid], timestamps[valid], diameters[valid])
    return result


# ---------------------------------------------------------------------------
# Multi-stimulus averaging
# ---------------------------------------------------------------------------


def average_plr_responses(
    responses: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average multiple PLR responses aligned to stimulus onset.

    Args:
        responses: List of diameter arrays, each aligned to stimulus onset.

    Returns:
        (mean_response, sem) — mean and standard error of the mean.
    """
    min_len = min(len(r) for r in responses)
    aligned = np.array([r[:min_len] for r in responses])
    mean = aligned.mean(axis=0)
    sem = aligned.std(axis=0, ddof=1) / np.sqrt(len(responses))
    return mean, sem


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------


def extract_plr_parameters(
    diameters: np.ndarray,
    timestamps: np.ndarray,
    stimulus_onset: float,
    confidence: np.ndarray | None = None,
    fps: float = 30.0,
    baseline_window: float = 0.5,
) -> PLRParameters:
    """Extract PLR biomarkers from a pupil diameter time series.

    Args:
        diameters: Pupil diameter per frame.
        timestamps: Timestamp per frame (seconds).
        stimulus_onset: Time of light stimulus (seconds).
        confidence: Optional per-frame segmentation confidence.
        fps: Frame rate for outlier rejection.
        baseline_window: Duration before stimulus to compute baseline (seconds).
    """
    diameters = diameters.astype(np.float64)
    timestamps = timestamps.astype(np.float64)

    # Compute baseline from pre-stimulus window
    baseline_mask = (timestamps >= stimulus_onset - baseline_window) & (timestamps < stimulus_onset)
    if baseline_mask.sum() < 2:
        baseline_mask = timestamps < stimulus_onset
    baseline = (
        float(np.median(diameters[baseline_mask])) if baseline_mask.any() else float(diameters[0])
    )

    # Preprocessing
    valid = detect_blinks(diameters, baseline, confidence=confidence)
    valid &= reject_outliers(diameters, fps)
    clean = interpolate_invalid(diameters, valid, timestamps)
    smoothed = smooth_plr(clean)

    # Fit parametric model to post-stimulus region
    fit_mask = timestamps >= stimulus_onset
    t_fit = timestamps[fit_mask]
    d_fit = smoothed[fit_mask]

    if len(t_fit) < 10:
        return PLRParameters(baseline, 0, 0, 0, 0, 0, 0)

    # Initial guesses
    d_min = d_fit.min()
    A_guess = max(baseline - d_min, 0.1)
    t_onset_guess = stimulus_onset + 0.2
    tau_c_guess = 0.15
    tau_d_guess = 1.0

    p0 = [baseline, A_guess, t_onset_guess, tau_c_guess, tau_d_guess]
    bounds = (
        [baseline * 0.5, 0.01, stimulus_onset, 0.01, 0.05],
        [baseline * 1.5, baseline, stimulus_onset + 1.0, 2.0, 10.0],
    )

    try:
        popt, _ = curve_fit(_plr_model, t_fit, d_fit, p0=p0, bounds=bounds, maxfev=5000)
        d0, A, t_onset, tau_c, tau_d = popt
    except RuntimeError:
        return PLRParameters(baseline, 0, 0, 0, 0, 0, 0)

    # Goodness of fit
    predicted = _plr_model(t_fit, *popt)
    ss_res = np.sum((d_fit - predicted) ** 2)
    ss_tot = np.sum((d_fit - d_fit.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pcv = _peak_constriction_velocity(A, tau_c, tau_d)

    return PLRParameters(
        baseline_diameter=d0,
        constriction_amplitude=A,
        constriction_amplitude_pct=A / d0 * 100.0 if d0 > 0 else 0.0,
        constriction_latency=t_onset - stimulus_onset,
        peak_constriction_velocity=pcv,
        recovery_time_constant=tau_d,
        fit_r_squared=r_squared,
    )


# ---------------------------------------------------------------------------
# Synthetic PLR generation (for testing / validation)
# ---------------------------------------------------------------------------


def generate_synthetic_plr(
    fps: float = 30.0,
    duration: float = 5.0,
    baseline: float = 6.0,
    amplitude: float = 2.0,
    latency: float = 0.2,
    tau_c: float = 0.15,
    tau_d: float = 1.5,
    noise_std: float = 0.1,
    pre_stimulus: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a synthetic PLR waveform with known parameters.

    Returns:
        (timestamps, diameters, stimulus_onset)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    total_duration = pre_stimulus + duration
    n_frames = int(total_duration * fps)
    timestamps = np.linspace(0, total_duration, n_frames)
    stimulus_onset = pre_stimulus

    diameters = _plr_model(
        timestamps,
        baseline,
        amplitude,
        stimulus_onset + latency,
        tau_c,
        tau_d,
    )
    diameters += rng.normal(0, noise_std, n_frames)

    return timestamps, diameters, stimulus_onset
