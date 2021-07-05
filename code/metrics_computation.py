import numpy as np


def compute_metrics(FAR, FRR, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    c_norm = np.zeros(len(thresholds))
    min_c_det_threshold = thresholds[0][0]
    beta = c_fa/c_miss + (1-p_target)/p_target
    for i in range(0, len(FRR)):
        c_det = c_miss * FRR[i] * p_target + c_fa * FAR[i] * (1 - p_target)
        c_norm[i] = FRR[i] + beta* FAR[i]

        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i] #[0][i]

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    c_prim = np.sum(c_norm)/len(c_norm)
    return min_dcf, min_c_det_threshold, c_prim


f = open("D:\licenta\LICENSE "
         "WORK\Speaker-Recognition\code\models\_thresholds_128.txt",
         "rb")
threshold_to_test = np.load(f)
f.close
f = open(r"D:\licenta\LICENSE "
         r"WORK\Speaker-Recognition\code\models\far_rates_128.txt",
         "rb")
FAR = np.load(f)
f.close
f = open(r"D:\licenta\LICENSE "
         r"WORK\Speaker-Recognition\code\models\frr_rates_128.txt",
         "rb")
FRR = np.load(f)
f.close
min_dcf, min_c_det_threshold, c_prim = compute_metrics(FAR,FRR,
                                                     threshold_to_test,
                                               0.068,1,1)

print("ready")