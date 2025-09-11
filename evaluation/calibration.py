"""Reliability-диаграмма для вероятностных голов."""
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_reliability(y_true, p_pred, n_bins=10, title="Reliability"):
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy='quantile')
    plt.figure()
    plt.plot([0,1],[0,1],'--', label='ideal')
    plt.plot(mean_pred, frac_pos, marker='o', label='model')
    plt.xlabel('Predicted probability'); plt.ylabel('Empirical frequency')
    plt.title(title); plt.legend(); plt.grid(True); plt.show()
