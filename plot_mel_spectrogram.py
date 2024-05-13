import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
path = "E:\\plot_mel_spec\\wavs"
speech_names = []
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if filename.lower().endswith(".wav"):
            # print(os.path.join(dirpath,filename))
            speech_names.append(os.path.join(dirpath, filename))

# plt.figure(1)
plt.figure(figsize=(20,20))
index = 0
for name in speech_names:
    index +=1

    y, sr = librosa.load(name, sr=16000)
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=256, n_mels=128)
    melspec = 10 * np.log10(melspec + 1e-8)


    plt.subplot(2, 5, index)
    librosa.display.specshow(melspec, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel spectrogram: %s' %os.path.basename(name))

plt.subplots_adjust(wspace=0.4, hspace=0.4)



# plt.figure(2)
plt.figure(figsize=(20,20))
index = 0
for name in speech_names:
    index +=1

    y, sr = librosa.load(name, sr=16000)
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=256, n_mels=128)
    melspec = 10 * np.log10(melspec + 1e-8)
    delta_1_mel = np.zeros_like(melspec)

    for i in range(2, delta_1_mel.shape[0]-3):
        delta_1_mel[i, :] = -2*melspec[i-2,:]-melspec[i-1,:]+melspec[i+1,:]+2*melspec[i+2,:]
    delta_1_mel = delta_1_mel / 3
    delta_2_mel =  np.zeros_like(delta_1_mel)

    plt.subplot(2, 5, index)
    librosa.display.specshow(delta_1_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.title('1st delta: %s' %os.path.basename(name))

plt.subplots_adjust(wspace=0.4, hspace=0.4)



# plt.figure(3)
plt.figure(figsize=(20,20))
index = 0
for name in speech_names:
    index +=1

    y, sr = librosa.load(name, sr=16000)
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=256, n_mels=128)
    melspec = 10 * np.log10(melspec + 1e-8)
    delta_1_mel = np.zeros_like(melspec)

    for i in range(2, delta_1_mel.shape[0]-3):
        delta_1_mel[i, :] = -2*melspec[i-2,:]-melspec[i-1,:]+melspec[i+1,:]+2*melspec[i+2,:]
    delta_1_mel = delta_1_mel / 3
    delta_2_mel =  np.zeros_like(delta_1_mel)

    for i in range(2, delta_2_mel.shape[0]-3):
        delta_2_mel[i, :] = -2*delta_1_mel[i-2,:]-delta_1_mel[i-1,:]+delta_1_mel[i+1,:]+2*delta_1_mel[i+2,:]
    delta_2_mel = delta_2_mel / 3

    plt.subplot(2, 5, index)
    librosa.display.specshow(delta_2_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.title('2nd delta: %s' %os.path.basename(name))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

plt.show()
