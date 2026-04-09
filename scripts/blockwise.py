import os
import re
import numpy as np
import mne
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "blockwise_output"

TASK_MODE = "8"

BLOCK_NAMES = [
    "learningBlock_1",
    "learningBlock_2",
    "learningBlock_3",
    "learningBlock_4",
    "learningBlock_5"
]

TMIN = -0.5
TMAX = 1.5
BASELINE = (-0.5, 0)

THETA_FREQS = np.arange(4, 8)   # 4,5,6,7 Hz
N_CYCLES = THETA_FREQS / 2.0

# Use your chosen frontal channels here
FALLBACK_FRONTAL = ["E3", "E4", "E9", "E10", "E11", "E15", "E16", "E18", "E19", "E22", "E23"]

MIN_TRIALS = 3


# =========================
# HELPERS
# =========================

def extract_subject_code(filename):
    m = re.search(r"(sub-[A-Za-z0-9]+)", filename)
    return m.group(1) if m else None

def file_matches_task_mode(filename, mode):
    if mode == "8":
        return "seqLearning8target" in filename
    elif mode == "6":
        return "seqLearning6target" in filename
    elif mode == "both":
        return ("seqLearning8target" in filename) or ("seqLearning6target" in filename)
    else:
        raise ValueError("TASK_MODE must be '8', '6', or 'both'.")

def get_frontal_channels(raw):
    frontal = []
    for short in FALLBACK_FRONTAL:
        for ch in raw.ch_names:
            if ch == short or ch.endswith(" " + short) or ch.endswith(short):
                frontal.append(ch)
                break
    return frontal


def crop_to_task(raw):
    start, stop = None, None
    for ann in raw.annotations:
        if ann["description"] == "seqLearning_start":
            start = ann["onset"]
        elif ann["description"] == "seqLearning_stop":
            stop = ann["onset"]
    if start is not None and stop is not None and stop > start:
        return raw.copy().crop(tmin=start, tmax=stop)
    return raw.copy()


def assign_trials_per_block(raw):
    anns = sorted([(a["onset"], str(a["description"])) for a in raw.annotations])
    trial_names = {f"dot_no{i}_ON" for i in range(1, 9)}

    current_block = None
    block_trials = {b: [] for b in BLOCK_NAMES}

    for onset, desc in anns:
        if desc in BLOCK_NAMES:
            current_block = desc
        elif desc in trial_names:
            if current_block is not None:
                block_trials[current_block].append(onset)

    return block_trials


# =========================
# CORE
# =========================

def compute_theta_metrics(raw, times, picks):
    """
    Returns:
    mean_theta, peak_theta_power, peak_theta_freq
    """
    if len(times) < MIN_TRIALS:
        return np.nan, np.nan, np.nan

    sfreq = raw.info["sfreq"]
    events = np.array([[int(t * sfreq), 0, 1] for t in times])

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"stim": 1},
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        picks=picks,
        reject_by_annotation=True,
        preload=True,
        verbose=False
    )

    if len(epochs) < MIN_TRIALS:
        return np.nan, np.nan, np.nan

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=THETA_FREQS,
        n_cycles=N_CYCLES,
        return_itc=False,
        average=True,
        verbose=False
    )

    # percentage baseline normalization
    power.apply_baseline(BASELINE, mode="percent")

    # use post-stimulus window only
    mask = (power.times >= 0.2) & (power.times <= 0.8)
    data = power.data[..., mask]   # channels x freqs x time

    # mean theta
    mean_theta = data.mean()

    # peak theta power and its frequency
    freq_power = data.mean(axis=(0, 2))   # average over channels and time
    peak_idx = np.argmax(freq_power)
    peak_theta_power = freq_power[peak_idx]
    peak_theta_freq = THETA_FREQS[peak_idx]

    return mean_theta, peak_theta_power, peak_theta_freq


def plot_blockwise(data, title, ylabel, filename, output_dir):
    x = np.arange(1, 6)

    plt.figure(figsize=(8, 6))

    # individual subjects
    for subj in data:
        plt.plot(x, subj, marker="o", alpha=0.35, linewidth=1)

    # mean + SEM
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(len(data))

    plt.plot(x, mean, marker="o", linewidth=3, label="Mean")
    plt.fill_between(x, mean - sem, mean + sem, alpha=0.2)
    plt.xticks(x, ["B1", "B2", "B3", "B4", "B5"])
    plt.xlabel("Learning Block")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_dir = os.path.join(base, INPUT_FOLDER)
    output_dir = os.path.join(base, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".edf")])

    all_mean = []
    all_peak_power = []
    all_peak_freq = []
    used_subjects = []

    for f in files:
        subj = extract_subject_code(f)
        if not subj:
            continue

        print(f"\nProcessing {subj}")

        raw = mne.io.read_raw_edf(os.path.join(input_dir, f), preload=True, verbose=False)
        raw = crop_to_task(raw)

        frontal = get_frontal_channels(raw)
        print("Frontal channels:", frontal)

        block_trials = assign_trials_per_block(raw)

        subj_mean = []
        subj_peak_power = []
        subj_peak_freq = []

        for block in BLOCK_NAMES:
            times = block_trials[block]
            mean_theta, peak_power, peak_freq = compute_theta_metrics(raw, times, frontal)

            subj_mean.append(mean_theta)
            subj_peak_power.append(peak_power)
            subj_peak_freq.append(peak_freq)

            print(f"{block}: trials={len(times)}, mean={mean_theta:.6e}, peak_power={peak_power:.6e}, peak_freq={peak_freq}")

        all_mean.append(subj_mean)
        all_peak_power.append(subj_peak_power)
        all_peak_freq.append(subj_peak_freq)
        used_subjects.append(subj)

    data_mean = np.array(all_mean)
    data_peak_power = np.array(all_peak_power)
    data_peak_freq = np.array(all_peak_freq)

    valid = ~np.isnan(data_mean).any(axis=1)
    data_mean = data_mean[valid]
    data_peak_power = data_peak_power[valid]
    data_peak_freq = data_peak_freq[valid]
    used_subjects = [s for i, s in enumerate(used_subjects) if valid[i]]

    print(f"\nUsed subjects: {len(used_subjects)}")

    # plots
    plot_blockwise(
        data_mean,
        "Mean Theta Across Learning Blocks",
        "Mean Theta Power (% change)",
        "blockwise_mean_theta.png",
        output_dir
    )

    plot_blockwise(
        data_peak_power,
        "Peak Theta Power Across Learning Blocks",
        "Peak Theta Power (% change)",
        "blockwise_peak_theta_power.png",
        output_dir
    )

    plot_blockwise(
        data_peak_freq,
        "Peak Theta Frequency Across Learning Blocks",
        "Peak Theta Frequency (Hz)",
        "blockwise_peak_theta_frequency.png",
        output_dir
    )

    # summary CSV
    summary_csv = os.path.join(output_dir, "blockwise_theta_summary.csv")
    with open(summary_csv, "w") as f:
        header = ["subject"]
        header += [f"mean_B{i}" for i in range(1, 6)]
        header += [f"peakpower_B{i}" for i in range(1, 6)]
        header += [f"peakfreq_B{i}" for i in range(1, 6)]
        f.write(",".join(header) + "\n")

        for i, subj in enumerate(used_subjects):
            row = [subj]
            row += [str(x) for x in data_mean[i]]
            row += [str(x) for x in data_peak_power[i]]
            row += [str(x) for x in data_peak_freq[i]]
            f.write(",".join(row) + "\n")

    # stats txt
    stats_txt = os.path.join(output_dir, "blockwise_theta_stats.txt")
    with open(stats_txt, "w") as f:
        f.write(f"N subjects: {len(data_mean)}\n\n")

        f.write("Mean theta across blocks:\n")
        for i in range(5):
            f.write(
                f"Block {i+1}: mean = {np.nanmean(data_mean[:, i]):.6e}, "
                f"sd = {np.nanstd(data_mean[:, i], ddof=1):.6e}\n"
            )
        f.write("\nPeak theta power across blocks:\n")
        for i in range(5):
            f.write(
                f"Block {i+1}: mean = {np.nanmean(data_peak_power[:, i]):.6e}, "
                f"sd = {np.nanstd(data_peak_power[:, i], ddof=1):.6e}\n"
            )

        f.write("\nPeak theta frequency across blocks:\n")
        for i in range(5):
            f.write(
                f"Block {i+1}: mean = {np.nanmean(data_peak_freq[:, i]):.6e}, "
                f"sd = {np.nanstd(data_peak_freq[:, i], ddof=1):.6e}\n"
            )

    print("\nDone.")