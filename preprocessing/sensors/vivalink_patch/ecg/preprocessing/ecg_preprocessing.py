from scipy import signal
from scipy.signal import medfilt

def fix_baseline_wander(data, fs=128):
    """
    BaselineWanderRemovalMedian adapted from ecg-kit.
    
    Given an array of amplitude values (data) and sample rate (fs),
    it applies two median filters to data to compute the baseline.
    The returned result is the original data minus this computed baseline.
    
    Args:
        data (numpy.ndarray): Array of amplitude values (ECG data).
        fs (int): Sample rate (Hz).
        
    Returns:
        numpy.ndarray: ECG data with baseline wander removed.
    """
    # Source: https://pypi.python.org/pypi/BaselineWanderRemoval/2017.10.25

    winsize = int(round(0.2 * fs))
    if winsize % 2 == 0:
        winsize += 1
    baseline_estimate = medfilt(data, kernel_size=winsize)
    
    winsize = int(round(0.6 * fs))
    if winsize % 2 == 0:
        winsize += 1
    baseline_estimate = medfilt(baseline_estimate, kernel_size=winsize)
    
    ecg_blr = data - baseline_estimate
    return ecg_blr

