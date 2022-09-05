import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image
import scipy.io.wavfile as wav
import librosa

"""
This script creates spectrogram matrices from wav files that can be passed \
to the CNN. This was heavily adopted from Frank Zalkow's work.
"""

class LogMelExtractor:
    """
    Creates a log-Mel Spectrogram of some input audio data. It first creates
    a mel filter and then applies the transformation of this mel filter to
    the STFT representation of the audio data

    Inputs
        sample_rate: int - The sampling rate of the original audio data
        window_size: int - The size of the window to be used for the mel
                     filter and the STFT transformation
        hop_size: int - The distance the window function will move over the
                  audio data - Related to the overlap = window_size - hop_size
        mel_bins: int - The number of bins for the mel filter
        fmin: int - The minimum frequency to start working from default=0
        fmax: int - The maximum frequency to start working from. Nyquist limit

    Output
        logmel_spectrogram: numpy.array - The log-Mel spectrogram
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, window_func, log=True, snv=True):
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = window_func
        self.log = log
        self.snv = snv

        # Output in the form of ((n_fft//2 + 1), mel_bins)
        self.melW = librosa.filters.mel(sr=sample_rate,
                                        n_fft=window_size,
                                        n_mels=mel_bins,
                                        fmin=fmin,
                                        fmax=fmax)

    def transform(self, audio_path):
        """
        Performs the transformation of the mel filter and the STFT
        representation of the audio data
        """
        # Compute short-time Fourier transform
        # Output in the form of (N, (n_fft//2 + 1))
        updated_file, sample_rate = librosa.load(audio_path, sr=None)
        stft_matrix = sepctrogram(audio=updated_file,
                                  window_size=self.window_size,
                                  hop_size=self.hop_size,
                                  squared=True,
                                  window_func=self.window_func)

        # Mel spectrogram
        spectrogram = np.dot(stft_matrix.T, self.melW.T)

        # Log mel spectrogram
        if self.log:
            spectrogram = librosa.core.power_to_db(spectrogram, ref=1.0,
                                                   amin=1e-10, top_db=None)

        spectrogram = spectrogram.astype(np.float32).T

        if self.snv:
            spectrogram = standard_normal_variate(spectrogram)

        return spectrogram


def sepctrogram(audio, window_size, hop_size, squared,
                window_func=np.hanning(1024), snv=False):
    """
    Computes the STFT of some audio data.

    Inputs
        audio: numpy.array - The audio data
        window_size: int - The size of the window passed over the data
        hop_size: int - The distance between windows
        squared: bool - If True, square the output matrix
        window_func: numpy.array - The window function to be passed over data
    """
    stft_matrix = librosa.core.stft(y=audio,
                                    n_fft=window_size,
                                    hop_length=hop_size,
                                    window=window_func,
                                    center=True,
                                    dtype=np.complex64,
                                    pad_mode='reflect')

    stft_matrix = np.abs(stft_matrix)
    if squared:
        stft_matrix = stft_matrix ** 2

    if snv:
        stft_matrix = standard_normal_variate(stft_matrix)

    return stft_matrix

def standard_normal_variate(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """
    Short-time Fourier transform of audio signal.
    """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.int(frameSize/2.0)), sig.astype(int)).astype(int)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples.astype(int), np.zeros(frameSize).astype(int)).astype("float64")

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames=frames.astype(float)
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically.
    """
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale)).astype(int)

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs


def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims)  # weird - not sure why it needs flipping
    ims = ims.astype(np.float32)
    if save_png:
        create_png(ims, png_name)

    return ims


def mfcc(audio_path,window_func=np.hanning(1024), snv=False):
    """
    Obtains the local differential (first and second order) of the MFCC

    Inputs
        audio: np.array - The audio data to be converted to MFCC
        sample_rate: int - The original sampling rate of the audio
        freq_bins: int - The number of mel bins
        window_size: int - The length of the window to be passed over the data
        hop_size: int - The gap between windows
        window: numpy - The type of window function to be used

    Output
        mfcc: numpy.array - The Updated MFCC
    """
    freq_bins = 40
    win_size = 1024
    hop_size = 512
    snv = True
    window_func = np.hanning(1024)
    updated_file, sample_rate = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=updated_file,
                                sr=sample_rate,
                                n_mfcc=freq_bins,
                                n_fft=win_size,
                                hop_length=hop_size,
                                window=window_func)

    if snv:
        mfcc = standard_normal_variate(mfcc)

    return mfcc

def create_png(im_matrix, png_name):
    """
    Save grayscale png of spectrogram.
    """
    image = Image.fromarray(im_matrix)
    image = image.convert('L')  # convert to grayscale
    image.save(png_name)



