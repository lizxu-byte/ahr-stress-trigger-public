"""Additional heart rate (AHR) estimation and simulated ESM triggering.

Modules
-------
io_wesad   Loader for the public WESAD dataset (wrist / Empatica E4 signals).
synthetic  Synthetic PPG + accelerometer generator used for tests and demos.
ppg        Filtering, peak detection, IBI cleaning, HR / HRV in windows.
sqi        PPG signal-quality indices (template correlation, spectral entropy).
motion     Accelerometer gravity removal and activity / motion flags.
pipeline   Glue: raw wrist signals -> per-window feature table.
ahr        Activity -> HR calibration model and additional heart rate.
trigger    Streaming ESM trigger simulator (past data only).
evaluate   Trigger metrics, threshold sweeps, LOSO tuning, mixed models.
io_bidmc   Loader for the PhysioNet BIDMC PPG/ECG recordings (WFDB format 16).
validation Beat matching, PPG-ECG delay, Bland-Altman agreement.
windows    Classification windows, calibration z-scoring, LOSO splits, metrics.
cnn        Small 1D-CNN and training loop (optional [dl] extra, PyTorch).
"""

__version__ = "0.1.0"
