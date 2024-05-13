import os, argparse
import numpy as np
import pandas as pd
import soundfile as sf
from librosa.core import stft, istft, magphase
from numpy.matlib import repmat
import soundfile
from pesq import pesq
from pystoi.stoi import stoi
def mhvals(d):
    dmh = np.array([[1,0.000,0.000],   [2,0.260,0.150],   [5,0.480,0.480],  [8,0.580,0.780],
                    [10,0.610,0.980],  [15,0.668,1.550],  [20,0.705,2.000], [30,0.762,2.300],
                    [40,0.800,2.520],  [60,0.841,3.100],  [80,0.865,3.38],  [120,0.890,4.150],
                    [140,0.900,4.350], [160,0.910,4.250], [180,0.920,3.90], [220,0.930,4.100],
                    [260,0.935,4.700], [300,0.940,5]])

    p = np.where(d<dmh[:,0])[0]
    if len(p)==0:
        i = dmh.shape[0]; j = i;
    else:
        i = p[0]
        j = i-1
    if d == dmh[i,0]:
        m = dmh[i,1]; h = dmh[i,2]
    else:
        qj = np.sqrt(dmh[i-1,0]); qi = np.sqrt(dmh[i,0]); q = np.sqrt(d);
        h  = dmh[i,2] + (q-qi) * (dmh[j,2]-dmh[i,2]) / (qj-qi)
        m  = dmh[i,1] + (qi*qj/q-qj) * (dmh[j,1]-dmh[i,1]) / (qi-qj)

    return m, h, d


def max_in_array(array, number):
    new_array = np.zeros(array.shape)
    for i in range(len(array)):
        if array[i] >= number:
            new_array[i] = array[i]
        else:
            new_array[i] = number
    return new_array


def min_in_array(array, number):
    new_array = np.zeros(array.shape)
    for i in range(len(array)):
        if array[i] <= number:
            new_array[i] = array[i]
        else:
            new_array[i] = number
    return new_array

def min_of_arrays(array1, array2):
    new_array = np.zeros(array1.shape)
    for i in range(len(array1)):
        if array1[i] <= array2[i]:
            new_array[i] = array1[i]
        else:
            new_array[i] = array2[i]
    return new_array

def max_of_arrays(array1, array2):
    new_array = np.zeros(array1.shape)
    for i in range(len(array1)):
        if array1[i] >= array2[i]:
            new_array[i] = array1[i]
        else:
            new_array[i] = array2[i]
    return new_array



def optimal_smoothing(magResponse):
    yf          = magResponse.transpose()**2
    (nr,nrf)    = yf.shape                             # number of frames and freq bins
    x           = np.zeros((nr,nrf))                   # initialize output arrays
    xs          = np.zeros((nr,nrf))                   # will hold std error in the future

    # Initializations
    tinc        = 0.008                                # second argument is frame increment
    nrcum       = 0                                    # no frames so far
    taca        = 0.0449                               # smoothing time constant for alpha_c = -tinc/log(0.7) in equ (11)
    tamax       = 0.392                                # max smoothing time constant in (3) = -tinc/log(0.96)
    taminh      = 0.0133                               # min smoothing time constant (upper limit) in (3) = -tinc/log(0.3)
    tpfall      = 0.064                                # time constant for P to fall (12)
    tbmax       = 0.0717                               # max smoothing time constant in (20) = -tinc/log(0.8)
    qeqmin      = 2                                    # minimum value of Qeq (23)
    qeqmax      = 14                                   # max value of Qeq per frame
    av          = 2.12                                 # fudge factor for bc calculation (23 + 13 lines)
    td          = 1.536                                # time to take minimum over
    nu          = 8                                    # number of subwindows
    qith        = np.array([0.03,0.05,0.06,np.Inf])    # noise slope thresholds in dB/s
    nsmdb       = np.array([47,31.4,15.7,4.1])
    pxx         = np.zeros((nr, nrf))
    alpha       = np.zeros((nr, nrf))

    # derived algorithm constants
    aca         = np.exp(-tinc/taca)                   # smoothing constant for alpha_c in equ (11) = 0.7
    acmax       = aca                                  # min value of alpha_c = 0.7 in equ (11) also = 0.7
    amax        = np.exp(-tinc/tamax)                  # max smoothing constant in (3) = 0.96
    aminh       = np.exp(-tinc/taminh)                 # min smoothing constant (upper limit) in (3) = 0.3
    bmax        = np.exp(-tinc/tbmax)                  # max smoothing constant in (20) = 0.8
    snrexp      = -tinc/tpfall
    nv          = np.round(td/(tinc*nu))               # length of each subwindow in frames

    if nv < 4:                                         # algorithm doesn't work for miniscule frames
        nv = 4
        nu = np.max([np.round(td/(tinc*nv)),1])

    nd        = nu*nv                                   # length of total window in frames
    md, hd, _ = mhvals(nd)                              # calculate the constants M(D) and H(D) from Table III
    mv, hv, _ = mhvals(nv)                              # calculate the constants M(D) and H(D) from Table III
    nsms      = 10**(nsmdb*nv*tinc/10)                  # [8 4 2 1.2] in paper
    qeqimax   = 1.0/qeqmin                              # maximum value of Qeq inverse (23)
    qeqimin   = 1.0/qeqmax                              # minumum value of Qeq per frame inverse

    if not nrcum:
        p          = yf[0,:]                            # smoothed power spectrum
        ac         = 1                                  # correction factor (9)
        sn2        = p                                  # estimated noise power
        pb         = p                                  # smoothed noisy speech power (20)
        pb2        = pb**2
        pminu      = p
        actmin     = np.array([np.Inf]*nrf)             # Running minimum estimate
        actminsub  = actmin                             # sub-window minimum estimate
        subwc      = nv                                 # force a buffer switch on first loop
        actbuf     = repmat(np.Inf,nu, nrf)             # buffer to store subwindow minima
        ibuf       = 0
        lminflag   = np.zeros((1,nrf)).flatten()        # flag to remember local minimum

    for t in range(nr):
        yft        = yf[t,:]                                    # noise speech power spectrum
        acb        = (1+(np.sum(p)/(np.sum(yft)-1)**2)**(-1) +1e-8)   # alpha_c-bar(t)  (9)
        ac         = aca*ac + (1-aca)*np.max([acb,acmax])       # alpha_c(t)  (10)
        ah         = (amax*ac) * (1+(p/(sn2+1e-8)-1)**2)**(-1)         # alpha_hat: smoothing factor per frequency (11)

        snr        = np.sum(p)/(np.sum(sn2)+1e-8)
        ah         = max_in_array(ah,np.min([aminh, (snr+1e-8)**(snrexp +1e-8)]))  # lower limit for alpha_hat (12)


        p          = ah*p + (1-ah)*yft            # smoothed noisy speech power (3)
        b          = max_in_array(ah**2, bmax)              # smoothing constant for estimating periodogram variance (22 + 2 lines)
        pb         = b*pb  + (1-b)*p            # smoothed periodogram (20)
        pb2        = b*pb2 + (1-b)*(p**2)       # smoothed periodogram squared (21)
        pxx[t,:]   = p
        alpha[t,:] = ah
        qeqi       = max_in_array(min_in_array((pb2-pb**2)/(2*sn2**2 +1e-8),qeqimax),qeqimin/(t+nrcum+1))   # Qeq inverse (23)
        qiav       = np.sum(qeqi)/nrf             # Average over all frequencies (23+12 lines) (ignore non-duplication of DC and nyquist terms)
        bc         = 1 + av*np.sqrt(qiav)             # bias correction factor (23+11 lines)
        bmind      = 1 + 2*(nd-1)*(1-md)/(qeqi**(-1) - 2*md+1e-8)      # we use the simplified form (17) instead of (15)
        bminv      = 1 + 2*(nv-1)*(1-mv)/(qeqi**(-1) - 2*mv+1e-8)      # same expression but for sub windows
        kmod       = bc*p*bmind < actmin        # Frequency mask for new minimum

        if kmod.any():
            true_idx            = list(np.where(kmod)[0])
            actmin[true_idx]    = bc*p[true_idx]*bmind[true_idx]
            actminsub[true_idx] = bc*p[true_idx]*bminv[true_idx]

        if (subwc > 1) and (subwc < nv):                        # middle of buffer - allow a local minimum:
            lminflag      = np.logical_or(lminflag, kmod)       # potential local minimum frequency bins
            pminu         = min_of_arrays(actminsub,pminu);
            sn2           = pminu;
        else:
            if subwc >= nv:
                ibuf              = (ibuf % nu)#        # increment actbuf storage pointer
                actbuf[ibuf,:]    = actmin      # save sub-window minimum
                pminu             = np.min(actbuf, axis=0)
                i                 = np.where(qiav<qith)[0]
                nsm               = nsms[i[0]]              # noise slope max

                lmin              = np.logical_and( np.logical_and(np.array(lminflag), np.logical_not(kmod)),
                                                    np.logical_and(np.array(actminsub < (nsm*pminu)), np.array(actminsub > pminu)) )

                if lmin.any():
                    true_index           = list(np.where(kmod)[0])
                    pminu[true_index]    = actminsub[true_index]
                    actbuf[:,true_index] = repmat(pminu[true_index],nu,1)

                lminflag = np.zeros(lminflag.shape)
                actmin   = np.Inf*np.ones(actmin.shape)
                subwc    = 0

        subwc      = subwc + 1
        x[t,:]     = sn2
        qisq       = np.sqrt(qeqi)
        # empirical formula for standard error based on Fig 15 of [2]
        xs[t,:]    = sn2*np.sqrt(0.266*(nd+100*qisq)*qisq/(1+0.005*nd+6/nd)/(0.5*qeqi**(-1)+nd-1))

    pxx   = np.abs(np.sqrt(pxx.transpose() +1e-8))
    pxx = np.nan_to_num(pxx)
    alpha = alpha.transpose()

    return  pxx


def optimal_smoothing_pre(magResponse):
    yf          = magResponse.transpose()**2
    (nr,nrf)    = yf.shape                             # number of frames and freq bins
    x           = np.zeros((nr,nrf))                   # initialize output arrays
    xs          = np.zeros((nr,nrf))                   # will hold std error in the future

    # Initializations
    tinc        = 0.008                                # second argument is frame increment
    nrcum       = 0                                    # no frames so far
    taca        = 0.0449                               # smoothing time constant for alpha_c = -tinc/log(0.7) in equ (11)
    tamax       = 0.392                                # max smoothing time constant in (3) = -tinc/log(0.96)
    taminh      = 0.0133                               # min smoothing time constant (upper limit) in (3) = -tinc/log(0.3)
    tpfall      = 0.064                                # time constant for P to fall (12)
    tbmax       = 0.0717                               # max smoothing time constant in (20) = -tinc/log(0.8)
    qeqmin      = 2                                    # minimum value of Qeq (23)
    qeqmax      = 14                                   # max value of Qeq per frame
    av          = 2.12                                 # fudge factor for bc calculation (23 + 13 lines)
    td          = 1.536                                # time to take minimum over
    nu          = 8                                    # number of subwindows
    qith        = np.array([0.03,0.05,0.06,np.Inf])    # noise slope thresholds in dB/s
    nsmdb       = np.array([47,31.4,15.7,4.1])
    pxx         = np.zeros((nr, nrf))
    alpha       = np.zeros((nr, nrf))

    # derived algorithm constants
    aca         = np.exp(-tinc/taca)                   # smoothing constant for alpha_c in equ (11) = 0.7
    acmax       = aca                                  # min value of alpha_c = 0.7 in equ (11) also = 0.7
    amax        = np.exp(-tinc/tamax)                  # max smoothing constant in (3) = 0.96
    aminh       = np.exp(-tinc/taminh)                 # min smoothing constant (upper limit) in (3) = 0.3
    bmax        = np.exp(-tinc/tbmax)                  # max smoothing constant in (20) = 0.8
    snrexp      = -tinc/tpfall
    nv          = np.round(td/(tinc*nu))               # length of each subwindow in frames

    if nv < 4:                                         # algorithm doesn't work for miniscule frames
        nv = 4
        nu = np.max([np.round(td/(tinc*nv)),1])

    nd        = nu*nv                                   # length of total window in frames
    md, hd, _ = mhvals(nd)                              # calculate the constants M(D) and H(D) from Table III
    mv, hv, _ = mhvals(nv)                              # calculate the constants M(D) and H(D) from Table III
    nsms      = 10**(nsmdb*nv*tinc/10)                  # [8 4 2 1.2] in paper
    qeqimax   = 1.0/qeqmin                              # maximum value of Qeq inverse (23)
    qeqimin   = 1.0/qeqmax                              # minumum value of Qeq per frame inverse

    if not nrcum:
        p          = yf[0,:]                            # smoothed power spectrum
        ac         = 1                                  # correction factor (9)
        sn2        = p                                  # estimated noise power
        pb         = p                                  # smoothed noisy speech power (20)
        pb2        = pb**2
        pminu      = p
        actmin     = np.array([np.Inf]*nrf)             # Running minimum estimate
        actminsub  = actmin                             # sub-window minimum estimate
        subwc      = nv                                 # force a buffer switch on first loop
        actbuf     = repmat(np.Inf,nu, nrf)             # buffer to store subwindow minima
        ibuf       = 0
        lminflag   = np.zeros((1,nrf)).flatten()        # flag to remember local minimum

    for t in range(nr):
        yft        = yf[t,:]                                    # noise speech power spectrum
        if np.sum(yft) ==0:
            yft = yft +1e-3
        acb        = (1+(np.sum(p)/np.sum(yft)-1)**2)**(-1)     # alpha_c-bar(t)  (9)
        ac         = aca*ac + (1-aca)*np.max([acb,acmax])       # alpha_c(t)  (10)
        ah         = (amax*ac) * (1+(p/(sn2 + 1e-8)-1)**2)**(-1)         # alpha_hat: smoothing factor per frequency (11)
        if np.sum(sn2) ==0:
            sn2 = sn2 +1e-3
        snr        = np.sum(p)/np.sum(sn2)
        ah         = max_in_array(ah,np.min([aminh, (snr+1e-8)**snrexp]))  # lower limit for alpha_hat (12)



        p          = ah*p + (1-ah)*yft            # smoothed noisy speech power (3)
        b          = max_in_array(ah**2, bmax)              # smoothing constant for estimating periodogram variance (22 + 2 lines)
        pb         = b*pb  + (1-b)*p            # smoothed periodogram (20)
        pb2        = b*pb2 + (1-b)*(p**2)       # smoothed periodogram squared (21)
        pxx[t,:]   = p
        alpha[t,:] = ah
        qeqi       = max_in_array(min_in_array((pb2-pb**2)/(2*sn2**2 +1e-8),qeqimax),qeqimin/(t+nrcum+1))   # Qeq inverse (23)
        qiav       = np.sum(qeqi)/(nrf + 1e-8)              # Average over all frequencies (23+12 lines) (ignore non-duplication of DC and nyquist terms)
        bc         = 1 + av*np.sqrt(qiav)             # bias correction factor (23+11 lines)
        bmind      = 1 + 2*(nd-1)*(1-md)/(qeqi**(-1) - 2*md +1e-8)      # we use the simplified form (17) instead of (15)
        bminv      = 1 + 2*(nv-1)*(1-mv)/(qeqi**(-1) - 2*mv +1e-8)      # same expression but for sub windows
        kmod       = bc*p*bmind < actmin        # Frequency mask for new minimum

        if kmod.any():
            true_idx            = list(np.where(kmod)[0])
            actmin[true_idx]    = bc*p[true_idx]*bmind[true_idx]
            actminsub[true_idx] = bc*p[true_idx]*bminv[true_idx]

        if (subwc > 1) and (subwc < nv):                        # middle of buffer - allow a local minimum:
            lminflag      = np.logical_or(lminflag, kmod)       # potential local minimum frequency bins
            pminu         = min_of_arrays(actminsub,pminu);
            sn2           = pminu;
        else:
            if subwc >= nv:
                ibuf              = (ibuf % nu)#        # increment actbuf storage pointer
                actbuf[ibuf,:]    = actmin      # save sub-window minimum
                pminu             = np.min(actbuf, axis=0)
                i                 = np.where(qiav<qith)[0]
                nsm               = nsms[i[0]]              # noise slope max

                lmin              = np.logical_and( np.logical_and(np.array(lminflag), np.logical_not(kmod)),
                                                    np.logical_and(np.array(actminsub < (nsm*pminu)), np.array(actminsub > pminu)) )

                if lmin.any():
                    true_index           = list(np.where(kmod)[0])
                    pminu[true_index]    = actminsub[true_index]
                    actbuf[:,true_index] = repmat(pminu[true_index],nu,1)

                lminflag = np.zeros(lminflag.shape)
                actmin   = np.Inf*np.ones(actmin.shape)
                subwc    = 0

        subwc      = subwc + 1
        x[t,:]     = sn2
        qisq       = np.sqrt(qeqi)
        # empirical formula for standard error based on Fig 15 of [2]
        xs[t,:]    = sn2*np.sqrt(0.266*(nd+100*qisq)*qisq/(1+0.005*nd+6/nd)/(0.5*qeqi**(-1)+nd-1))

    pxx   = np.abs(np.sqrt(pxx.transpose()))
    alpha = alpha.transpose()

    return  pxx

def norm(x):
    return x/np.max(np.abs(x.flatten()))

def mag2dB(x):
    return 20*np.log10(np.abs(x)+np.spacing(1))

def dB2mag(x):
    return 10**(x/20)

def psd(audio, preprocess=False):
    audioSTFT = stft(audio, n_fft=512, hop_length=128, win_length=512)[:-1, :]
    Mag, Phase = np.abs(audioSTFT), np.angle(audioSTFT)
    nframes = int(256 * np.ceil(np.shape(Mag)[1] / 256))
    pad_size = nframes - np.shape(Mag)[1]
    variance = (np.mean(Mag[:10]) if np.mean(Mag[:10]) < 0.01 else 0.01)
    pad_seq = variance * np.random.randn(256, pad_size)
    Mag = np.hstack((Mag, pad_seq))
    Phase = np.hstack((Phase, 0.0 * pad_seq))

    if preprocess:
        Mag_smooth = mag2dB(norm(optimal_smoothing_pre(Mag)))
        Mag_smooth[Mag_smooth < -120] = -120
        minmax_smooth = [np.min(Mag_smooth), np.max(Mag_smooth)]
        Mag_smooth_norm = np.interp(Mag_smooth, minmax_smooth, [-1, 1])

    Mag = mag2dB(norm(Mag))
    Mag[Mag < -120] = -120
    minmax = [np.min(Mag), np.max(Mag)]
    Mag_norm = np.interp(Mag, minmax, [-1, 1])

    psd = {}
    if preprocess:
        psd['MagdB_smooth'] = Mag_smooth_norm
        psd['Norm_smooth'] = minmax_smooth
    psd['MagdB'] = Mag_norm
    psd['Phase'] = Phase
    psd['Norm'] = minmax
    return psd

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

if __name__ == '__main__':

    reverb_path = "D:\\DNS_NEW\\DNS_Challenge\\datasets\\training_set_DNS2_facebook_dereverb\\noisy"
    clean_path = "E:\\real_time_exp\\CRN_16k_DNS_2_dereverbonly\\test_for_IAM\\clean"
    speech_names = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(reverb_path):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                speech_names.append(os.path.join(dirpath, filename))
    for speech_na in speech_names:
        name = os.path.basename(speech_na)

        (x, fs) = soundfile.read(speech_na, dtype="float32")
        # (y, fs1) = soundfile.read("D:\deep learning\PSMtest\p226_010.FactoryFloor1.wav")
        # (y, fs1) = soundfile.read("D:\deep learning\PSMtest\p226_010clean.wav")
        sr = fs
        wav_data = x
        # wav_data = wav_data.astype(np.float32)

        psd_x = psd(wav_data, preprocess=True)

        if True in np.isnan(psd_x['MagdB_smooth']):
            print(name)
        enhanced_mag = psd_x['MagdB']

        enhanced_mag = np.interp(enhanced_mag, [-1, 1], psd_x['Norm'])
        temp = np.zeros((257, enhanced_mag.shape[1])) + 1j * np.zeros((257, enhanced_mag.shape[1]))
        temp[:-1, :] = 10 ** (enhanced_mag / 20) * (np.cos(psd_x['Phase']) + np.sin(psd_x['Phase']) * 1j)
        enhanced_audio = istft(temp)
        enhanced_audio = 0.98 * enhanced_audio / np.max(np.abs(enhanced_audio))

        enhanced_audio = enhanced_audio[0:len(wav_data)]

        if count % 100 == 0:
            print(count)
        count += 1

        # pesq_score = pesq(16000, wav_data, enhanced_audio, 'wb')
        # print("PESQ : %f" %pesq_score)
        #
        # stoi_score = stoi(wav_data, enhanced_audio, 16000, extended=False)
        # print("STOI : %f" %stoi_score)

        # workspace="D:\\real_time_exp\\SOTA_code\\SkipConvNet-master\\test_wavs"
        # out_audio_path = os.path.join(workspace, "p226_010_recons.wav")
        # create_folder(os.path.dirname(out_audio_path))
        # write_audio(out_audio_path, enhanced_audio, fs)