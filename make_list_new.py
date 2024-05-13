import os
import librosa
import soundfile as sf
path1 = 'D:\\JMCheng\\RT_16k\\DATASET\\vae_dns\\'

wav_names1 = os.listdir(path1)



f = open('D:\\JMCheng\\RT_16k\\16k_exp\\vae_dns_noisy80.lst', 'w')
for i in range(len(wav_names1)):
    speech_basename = os.path.basename(wav_names1[i])
    speech_fpart = os.path.splitext(speech_basename)[0]
    name_list = speech_fpart.split("_")
    if name_list[1] == "nearend":
        noisy, fs = sf.read(path1 + wav_names1[i])

        target_name = name_list[0] + "_" + "target" + ".wav"

        # clean, fs1 = sf.read(path1 + target_name)

        duration = len(noisy) / fs
        f.write(path1 + wav_names1[i] + ' ' + path1 + target_name + ' ' + '{}'.format(duration) + '\n')
        print("No.{} ".format(i) + wav_names1[i])
f.close()