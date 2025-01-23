"""Step counting method for accelerometer data.
adapted form: https://github.com/onnela-lab/forest/blob/develop/forest/oak/base.py
"""

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from ssqueezepy import ssq_cwt


def preprocess_bout(t_bout: np.ndarray, x_bout: np.ndarray, y_bout: np.ndarray,
                    z_bout: np.ndarray, fs: int = 25) -> tuple:
    """Preprocesses accelerometer bout to a common format.

    Resample 3-axial input signal to a predefined sampling rate and compute
    vector magnitude.

    Args:
        t_bout: array of floats
            Unix timestamp
        x_bout: array of floats
            X-axis acceleration
        y_bout: array of floats
            Y-axis acceleration
        z_bout: array of floats
            Z-axis acceleration
        fs: integer
            sampling frequency

    Returns:
        Tuple of ndarrays:
            - t_bout_interp: resampled timestamp (in Unix)
            - vm_bout_interp: vector magnitude of acceleration
    """
    t_bout_interp = t_bout - t_bout[0]
    t_bout_interp = np.arange(t_bout_interp[0], t_bout_interp[-1], (1/fs))
    t_bout_interp = t_bout_interp + t_bout[0]

    f = interpolate.interp1d(t_bout, x_bout)
    x_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, y_bout)
    y_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, z_bout)
    z_bout_interp = f(t_bout_interp)

    # adjust bouts using designated function
    x_bout_interp = adjust_bout(x_bout_interp)
    y_bout_interp = adjust_bout(y_bout_interp)
    z_bout_interp = adjust_bout(z_bout_interp)

    # number of full seconds of measurements
    num_seconds = np.floor(len(x_bout_interp)/fs)

    # trim and decimate t
    #t_bout_interp = t_bout_interp[:int(num_seconds*fs)]
    #t_bout_interp = t_bout_interp[::fs]

    # calculate vm
    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 +
                             z_bout_interp**2)

    # standardize measurement to gravity units (g) if its recorded in m/s**2
    if np.mean(vm_bout_interp) > 5:
        x_bout_interp = x_bout_interp/9.80665
        y_bout_interp = y_bout_interp/9.80665
        z_bout_interp = z_bout_interp/9.80665

    # calculate vm after unit verification
    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 +
                             z_bout_interp**2) - 1

    return t_bout_interp, vm_bout_interp, x_bout_interp, y_bout_interp, z_bout_interp


def adjust_bout(inarray: np.ndarray, fs: int = 25) -> np.ndarray:
    """Fills observations in incomplete bouts.

    For example, if the bout is 9.8s long, add values at its end to make it
    10s (results in N%fs=0).

    Args:
        inarray: array of floats
            input with one bout of activity
        fs: integer
            sampling frequency

    Returns:
        Ndarray with length-adjusted vector magnitude
    """
    # if data is available for 70% of the last second
    if len(inarray) % fs >= 0.7*fs:
        for i in range(fs-len(inarray) % fs):
            inarray = np.append(inarray, inarray[-1])
    # otherwise, trim the data to the full second
    else:
        inarray = inarray[np.arange(len(inarray)//fs*fs)]

    return inarray


def get_pp(vm_bout: np.ndarray, fs: int = 25) -> npt.NDArray[np.float64]:
    """Calculate peak-to-peak metric in one-second time windows.

    Args:
        vm_bout: array of floats
            vector magnitude with one bout of activity (in g)
        fs: integer
            sampling frequency (in Hz)

    Returns:
        Ndarray with metric

    """
    vm_res_sec = vm_bout.reshape((fs, -1), order="F")
    pp = np.array([max(vm_res_sec[:, i])-min(vm_res_sec[:, i])
                   for i in range(vm_res_sec.shape[1])])

    return pp


def compute_interpolate_cwt(tapered_bout: np.ndarray, fs: int = 25,
                            wavelet: tuple = ('gmw', {'beta': 90,
                                                      'gamma': 3})) -> tuple:
    """Compute and interpolate CWT over acceleration data.

    Args:
        tapered_bout: array of floats
            vector magnitude with one bout of activity (in g)
        fs: integer
            sampling frequency (in Hz)
        wavelet: tuple
            mother wavelet used to compute CWT

    Returns:
        Tuple of ndarrays with interpolated frequency and wavelet coefficients
    """
    # smooth signal on the edges to minimize impact of coin of influence
    window = tukey(len(tapered_bout), alpha=0.02, sym=True)
    tapered_bout = np.concatenate((np.zeros(5*fs),
                                   tapered_bout*window,
                                   np.zeros(5*fs)))

    # compute cwt over bout
    out = ssq_cwt(tapered_bout[:-1], wavelet, fs=25)
    coefs = out[0]
    coefs = np.append(coefs, coefs[:, -1:], 1)

    # magnitude of cwt
    coefs = np.abs(coefs**2)

    # interpolate coefficients
    freqs = out[2]
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    ip = interpolate.interp2d(range(coefs.shape[1]), freqs, coefs)
    coefs_interp = ip(range(coefs.shape[1]), freqs_interp)

    # trim spectrogram from the coi
    coefs_interp = coefs_interp[:, 5*fs:-5*fs]

    return freqs_interp, coefs_interp


def identify_peaks_in_cwt(freqs_interp: np.ndarray, coefs_interp: np.ndarray,
                          fs: int = 25, step_freq: tuple = (1.4, 2.3),
                          alpha: float = 0.6, beta: float = 2.5):
    """Identify dominant peaks in wavelet coefficients.

    Method uses alpha and beta parameters to identify dominant peaks in
    one-second non-overlapping windows in the product of Continuous Wavelet
    Transformation. Dominant peaks need tooccur within the step frequency
    range.

    Args:
        freqs_interp: array of floats
            frequency-domain (in Hz)
        coefs_interp: array of floats
            wavelet coefficients (-)
        fs: integer
            sampling frequency (in Hz)
        step_freq: tuple
            step frequency range
        alpha: float
            maximum ratio between dominant peak below and within
            step frequency range
        beta: float
            maximum ratio between dominant peak above and within
            step frequency range

    Returns:
        Ndarray with dominant peaks
    """
    # identify dominant peaks within coefficients
    dp = np.zeros((coefs_interp.shape[0], int(coefs_interp.shape[1]/fs)))
    loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
    loc_max = np.argmin(abs(freqs_interp-step_freq[1]))
    for i in range(int(coefs_interp.shape[1]/fs)):
        # segment measurement into one-second non-overlapping windows
        x_start = i*fs
        x_end = (i + 1)*fs
        # identify peaks and their location in each window
        window = np.sum(coefs_interp[:, np.arange(x_start, x_end)], axis=1)
        locs, _ = find_peaks(window)
        pks = window[locs]
        ind = np.argsort(-pks)
        locs = locs[ind]
        pks = pks[ind]
        index_in_range = []

        # account peaks that satisfy condition
        for j in range(len(locs)):
            if loc_min <= locs[j] <= loc_max:
                index_in_range.append(j)
            if len(index_in_range) >= 1:
                break
        peak_vec = np.zeros(coefs_interp.shape[0])
        if len(index_in_range) > 0:
            if locs[0] > loc_max:
                if pks[0]/pks[index_in_range[0]] < beta:
                    peak_vec[locs[index_in_range[0]]] = 1
            elif locs[0] < loc_min:
                if pks[0]/pks[index_in_range[0]] < alpha:
                    peak_vec[locs[index_in_range[0]]] = 1
            else:
                peak_vec[locs[index_in_range[0]]] = 1
        dp[:, i] = peak_vec

    return dp


def find_walking(vm_bout: np.ndarray, fs: int = 25, min_amp: float = 0.3,
                 step_freq: tuple = (1.4, 3.3), alpha: float = 0.6,
                 beta: float = 2.5, min_t: int = 3,
                 delta: int = 20) -> npt.NDArray[np.float64]:
    """Finds walking and calculate steps from raw acceleration data.

    Method finds periods of repetitive and continuous oscillations with
    predominant frequency occurring within know step frequency range.
    Frequency components are extracted with Continuous Wavelet Transform.

    Args:
        vm_bout: array of floats
            vector magnitude with one bout of activity (in g)
        fs: integer
            sampling frequency (in Hz)
        min_amp: float
            minimum amplitude (in g)
        step_freq: tuple
            step frequency range
        alpha: float
            maximum ratio between dominant peak below and within
            step frequency range
        beta: float
            maximum ratio between dominant peak above and within
            step frequency range
        min_t: integer
            minimum duration of peaks (in seconds)
        delta: integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        Ndarray with identified number of steps per second
    """
    # define wavelet function used in method
    wavelet = ('gmw', {'beta': 90, 'gamma': 3})

    # calculate peak-to-peak
    pp = get_pp(vm_bout, fs)

    # assume the entire bout is of high-intensity
    valid = np.ones(len(pp), dtype=bool)

    # exclude low-intensity periods
    valid[pp < min_amp] = False

    # compute cwt only if valid fragment is sufficiently long
    if sum(valid) >= min_t:
        # trim bout to valid periods only
        tapered_bout = vm_bout[np.repeat(valid, fs)]

        # compute and interpolate CWT
        freqs_interp, coefs_interp = compute_interpolate_cwt(tapered_bout, fs,
                                                             wavelet)

        # get map of dominant peaks
        dp = identify_peaks_in_cwt(freqs_interp, coefs_interp, fs, step_freq,
                                   alpha, beta)

        # distribute local maxima across valid periods
        valid_peaks = np.zeros((dp.shape[0], len(valid)))
        valid_peaks[:, valid] = dp

        # find peaks that are continuous in time (min_t) and frequency (delta)
        cont_peaks = find_continuous_dominant_peaks(valid_peaks, min_t, delta)

        # summarize the results
        cad = np.zeros(valid_peaks.shape[1])
        for i in range(len(cad)):
            ind_freqs = np.where(cont_peaks[:, i] > 0)[0]
            if len(ind_freqs) > 0:
                cad[i] = freqs_interp[ind_freqs[0]]

    else:
        cad = np.zeros(int(vm_bout.shape[0]/fs))

    return cad


def find_continuous_dominant_peaks(valid_peaks: np.ndarray, min_t: int,
                                   delta: int) -> npt.NDArray[np.float64]:
    """Identifies continuous and sustained peaks within matrix.

    Args:
        valid_peaks: nparray
            binary matrix (1=peak,0=no peak) of valid peaks
        min_t: integer
            minimum duration of peaks (in seconds)
        delta: integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        Ndarray with binary matrix (1=peak,0=no peak) of continuous peaks
    """
    valid_peaks = np.concatenate((valid_peaks,
                                  np.zeros((valid_peaks.shape[0], 1))), axis=1)
    cont_peaks = np.zeros((valid_peaks.shape[0], valid_peaks.shape[1]))
    for slice_ind in range(valid_peaks.shape[1] - min_t):
        slice_mat = valid_peaks[:, np.arange(slice_ind, slice_ind + min_t)]
        windows = ([i for i in np.arange(min_t)] +
                   [i for i in np.arange(min_t-2, -1, -1)])
        for win_ind in windows:
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            count = 0
            if len(pr) > 0:
                for i in range(len(pr)):
                    index = np.arange(max(0, pr[i] - delta),
                                      min(pr[i] + delta + 1,
                                          slice_mat.shape[0]
                                          ))
                    if win_ind == 0 or win_ind == min_t - 1:
                        cur_peak_loc = np.transpose(np.array(
                            [np.ones(len(index))*pr[i], index], dtype=int
                            ))
                    else:
                        cur_peak_loc = np.transpose(np.array(
                            [index, np.ones(len(index))*pr[i], index],
                            dtype=int
                            ))

                    peaks = np.zeros((cur_peak_loc.shape[0],
                                      cur_peak_loc.shape[1]), dtype=int)
                    if win_ind == 0:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0],
                                                win_ind]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1],
                                                win_ind + 1]
                    elif win_ind == min_t - 1:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0],
                                                win_ind]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1],
                                                win_ind - 1]
                    else:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0],
                                                win_ind - 1]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1],
                                                win_ind]
                        peaks[:, 2] = slice_mat[cur_peak_loc[:, 2],
                                                win_ind + 1]

                    cont_peaks_edge = cur_peak_loc[np.sum(
                        peaks[:, np.arange(2)], axis=1) > 1, :]
                    cpe0 = cont_peaks_edge.shape[0]
                    if win_ind == 0 or win_ind == min_t - 1:  # first or last
                        if cpe0 == 0:
                            slice_mat[cur_peak_loc[:, 0], win_ind] = 0
                        else:
                            count = count + 1
                    else:
                        cont_peaks_other = cur_peak_loc[np.sum(
                            peaks[:, np.arange(1, 3)], axis=1) > 1, :]
                        cpo0 = cont_peaks_other.shape[0]
                        if cpe0 == 0 or cpo0 == 0:
                            slice_mat[cur_peak_loc[:, 1], win_ind] = 0
                        else:
                            count = count + 1
            if count == 0:
                slice_mat = np.zeros((slice_mat.shape[0], slice_mat.shape[1]))
                break
        cont_peaks[:, np.arange(
            slice_ind, slice_ind + min_t)] = np.maximum(
                cont_peaks[:, np.arange(slice_ind, slice_ind + min_t)],
                slice_mat)

    return cont_peaks[:, :-1]


