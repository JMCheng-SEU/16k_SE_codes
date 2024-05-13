import librosa
import matplotlib.pyplot as plt
import librosa.display
import torch

flag = torch.cuda.is_available()
print(flag)

# wave_path = "E:\\real_time_exp\\CRN_16k\\16k_test_new\\recording_by_liang.wav"
# y, sr = librosa.load(wave_path, sr=16000)
# fig, ax = plt.subplots(2, 1)
#
# librosa.display.waveplot(y, sr=sr, label='test_wave', ax=ax[0])
# mag, _ = librosa.magphase(librosa.stft(y, n_fft=512, hop_length=256, win_length=512))
# librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=ax[1], sr=16000)
# plt.tight_layout()
# plt.show()

