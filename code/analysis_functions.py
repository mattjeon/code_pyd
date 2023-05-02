#신호에서 추출할 수 있는 생체 정보 구하는 함수들
#평균 맥박수, rppg, spo2, hrv 구하는 함수

import numpy as np
from hrvanalysis import get_time_domain_features,get_frequency_domain_features
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import zscore
import functions

#평균 맥박수 함수
def estimate_average_pulserate(data, fs, minBPM, maxBPM):
    # 평균 맥박수 측정
    f, pxx = signal.periodogram(data, fs=fs, window='hann')
    max_peak_idx = np.argmax(pxx)
    bpm = int(f[max_peak_idx] * 60)
    return min(max(bpm, minBPM), maxBPM)

#spo2 추출 함수
def rspo2_extract(rgb_signal, fs, minBPM, maxBPM, rspo2_buffer):
    r_crop = np.array(rgb_signal[0]) / 255
    g_crop = np.array(rgb_signal[1]) / 255
    b_crop = np.array(rgb_signal[2]) / 255

    # y = ((65.481 * r_crop) + (128.553 * g_crop) + (24.966 * b_crop)) + 16
    cropped_cg = ((-81.085 * r_crop) + (112 * g_crop) + (-30.915 * b_crop)) + 128
    cropped_cr = ((112 * r_crop) + (-93.786 * g_crop) + (-18.214 * b_crop)) + 128

    cr_pass = functions.temporal_bandpass_filter(cropped_cr, fs, minBPM, maxBPM)
    cg_pass = functions.temporal_bandpass_filter(cropped_cg, fs, minBPM, maxBPM)

    window_time = 10  # 10초씩 분석
    window_size = window_time * fs
    # ac 구하기
    step = len(cr_pass) - window_size + 1
    ac_cr = []
    ac_cg = []

    for i in range(step):
        cr_p = cr_pass[i:i + window_size]
        cr_v = -cr_pass[i:i + window_size]  # valley를 검출하기 위해서 신호 반전시킴
        cg_p = cg_pass[i:i + window_size]
        cg_v = -cg_pass[i:i + window_size]
        cr_peaks, _ = find_peaks(cr_p, distance=15)  # peak 검출
        cr_valleys, _ = find_peaks(cr_v, distance=15)  # valley 검출
        cg_peaks, _ = find_peaks(cg_p, distance=15)  # peak 검출
        cg_valleys, _ = find_peaks(cg_v, distance=15)  # valley 검출

        # peak와 valley의 개수가 같지 않을 수 있어서 최소 개수 지정
        cr_length = len(cr_peaks) if len(cr_peaks) < len(cr_valleys) else len(cr_valleys)
        cg_length = len(cg_peaks) if len(cg_peaks) < len(cg_valleys) else len(cg_valleys)

        cr_peak2valley = []
        cg_peak2valley = []

        # valley 나누기 peak
        for r in range(cr_length):
            ampl = np.abs(cr_p[cr_valleys[r]] / cr_p[cr_peaks[r]])
            cr_peak2valley.append(ampl)

        for g in range(cg_length):
            ampl = np.abs(cg_p[cg_valleys[g]] / cg_p[cg_peaks[g]])
            cg_peak2valley.append(ampl)

        # 중앙값 추출
        # 이는 노이즈를 막기 위해 중앙값 선택
        ac_cr.append(np.median(cr_peak2valley))
        ac_cg.append(np.median(cg_peak2valley))

    cr_log = np.log(np.array(ac_cr) + 1)
    cg_log = np.log(np.array(ac_cg) + 1)
    ratio = cr_log / cg_log
    # 선형회귀
    rspo2 = int(19.8805 * ratio + 72.9847)

    rspo2_buffer.append(rspo2)

    plot_rspo2 = None
    if len(rspo2_buffer) >= 20:
        avg_spo2 = int(np.average(rspo2_buffer))
        if avg_spo2 >= 100:
            avg_spo2 = 99
        plot_rspo2 = avg_spo2
        # fps : 30이라서
        if len(rspo2_buffer) > 30:
            del rspo2_buffer[0]

    return plot_rspo2

#hrv 분석 함수
def hrv_analysis(signal, times, sr=30, distance=100):
    th=2 # PPI normalize 임계값
    rppg_sig_ = np.array(signal, dtype='float32')
    rppg_time_ = np.array(times, dtype='float32')  # msec단위
    # ================= preprocessing===================
    # 1. interpolation
    rppg_time = np.linspace(rppg_time_[0], rppg_time_[-1], len(rppg_time_) * 8)  # interpolation된 x , 기존 cppg sr=255로 rppg의 약 8.5배
    i = interp1d(rppg_time_, rppg_sig_, kind='quadratic')
    rppg_sig = i(rppg_time)  # interpolation된 y

    sr=sr*8
    # 2. bandpass filtering
    filtered = functions.preprocessing(rppg_sig, 2.0, 0.5, sr)
    r_peaks_y, r_peaks_x = functions.detect_peak_hrv(rppg_time, filtered, distance)

    # 3. extract PPI
    rppg_ppi = np.diff(r_peaks_x)

    # 4. normalize PPI
    rppg_nni = rppg_ppi.copy()
    rppg_nni[np.abs(zscore(rppg_ppi)) > th] = np.median(rppg_ppi) # z_score=평균분포에 떨어진 정도,
    # ================ hrv analysis ======================
    t_hrv_feature = get_time_domain_features(rppg_nni)  # rppg_ppi r_nni
    f_hrv_feature = get_frequency_domain_features(rppg_nni)
    return t_hrv_feature,f_hrv_feature
