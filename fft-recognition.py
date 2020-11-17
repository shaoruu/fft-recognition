import librosa
import numpy as np
import scipy
import time

# Constants
LOADING_BAR = [
    "[=     ]",
    "[ =    ]",
    "[  =   ]",
    "[   =  ]",
    "[    = ]",
    "[     =]",
    "[    = ]",
    "[   =  ]",
    "[  =   ]",
    "[ =    ]",
]


def get_variance(a, b):
    # Helper function
    if len(a) != len(b):
        return False
    output = 0
    for i in range(len(a)):
        output += np.abs(np.abs(a[i])-np.abs(b[i]))
    return output


# Resource paths
full_clip_path = 'resources/full-clip.wav'
partial_clip_path = 'resources/clip1.wav'

# Audio data
fc_duration = librosa.get_duration(filename=full_clip_path)
pc_duration = librosa.get_duration(filename=partial_clip_path)

# Loading in audio
partial_audio, sampling_rate = librosa.load(
    partial_clip_path, sr=None, mono=True, offset=0.0, duration=None)
partial_fft = scipy.fft.fft(partial_audio)

# Tracking variables
min_variance = float('inf')
time_pos = -1
start_time = time.time()

# Loop through entire duration to find closest match
for i in range(0, round(fc_duration-pc_duration)):
    print(LOADING_BAR[i % len(LOADING_BAR)] + ' Elapsed: ' +
          str(round(time.time() - start_time, 2)) + 's', end="\r")

    full_audio, sampling_rate = librosa.load(
        full_clip_path, sr=None, mono=True, offset=i, duration=pc_duration)
    full_fft = scipy.fft.fft(full_audio)

    variance = get_variance(partial_fft, full_fft)
    if variance and variance < min_variance:
        min_variance = variance
        time_pos = i

print(f'The partial clip occurs at {time_pos} seconds.')
print(
    f'Entire audio search took {round(time.time() - start_time, 4)} seconds.')
