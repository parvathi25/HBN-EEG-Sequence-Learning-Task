import os
import re
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon


# =========================================================
# CONFIG
# =========================================================


INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "earlyvslate_output"


# Choose which sequence task to analyze:
# "8" -> only seqLearning8target
# "6" -> only seqLearning6target
# "both" -> both
TASK_MODE = "8"


# Correct block names for your EDF annotations
EARLY_BLOCKS = {"learningBlock_1", "learningBlock_2"}
LATE_BLOCKS = {"learningBlock_4", "learningBlock_5"}


# Epoch settings
TMIN = -0.5
TMAX = 1.5
BASELINE = (-0.5, 0.0)


# Theta settings
THETA_FREQS = np.arange(4, 8)   # 4,5,6,7 Hz
N_CYCLES = THETA_FREQS / 2.0


# Fixed frontal channels to use in all subjects
FALLBACK_FRONTAL = ["EEG E3", "EEG E4", "EEG E9", "EEG E10", "EEG E11", "EEG E15", "EEG E16", "EEG E18", "EEG E19", "EEG E22", "EEG E23"]
N_FRONTAL_CHANNELS = 11


MIN_TRIALS_PER_CONDITION = 3


# =========================================================
# HELPERS (updated frontal channel logic)
# =========================================================


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


# Use only the FALLBACK_FRONTAL channels; no 3D‑based selection
def get_frontal_channels(raw, n_channels=None):
    picks = mne.pick_channels(raw.ch_names, include=FALLBACK_FRONTAL)
    if len(picks) == 0:
        raise ValueError("None of the FALLBACK_FRONTAL channels exist in raw.")

    frontal_chs = [raw.ch_names[p] for p in picks]

    # Optionally keep only the first N_FRONTAL_CHANNELS keys from FALLBACK_FRONTAL that exist
    if len(frontal_chs) > N_FRONTAL_CHANNELS:
        frontal_chs = frontal_chs[:N_FRONTAL_CHANNELS]
    elif len(frontal_chs) < N_FRONTAL_CHANNELS:
        print(f"WARNING: only {len(frontal_chs)} frontal channels available (expected {N_FRONTAL_CHANNELS}).")

    return frontal_chs


def crop_to_task(raw):
    start = None
    stop = None

    for ann in raw.annotations:
        if ann["description"] == "seqLearning_start":
            start = ann["onset"]
        elif ann["description"] == "seqLearning_stop":
            stop = ann["onset"]

    if start is not None and stop is not None and stop > start:
        return raw.copy().crop(tmin=start, tmax=stop)

    return raw.copy()


def parse_annotations(raw):
    anns = []
    for ann in raw.annotations:
        anns.append({
            "onset": ann["onset"],
            "description": str(ann["description"])
        })
    anns = sorted(anns, key=lambda x: x["onset"])
    return anns


def get_trial_on_event_names():
    return {f"dot_no{i}_ON" for i in range(1, 9)}


def assign_trials_to_blocks(raw):
    """
    Assign each dot_noX_ON event to the most recent learningBlock_X.
    """
    anns = parse_annotations(raw)
    trial_names = get_trial_on_event_names()

    current_block = None
    early_trial_times = []
    late_trial_times = []

    for ann in anns:
        desc = ann["description"]
        onset = ann["onset"]

        if desc.startswith("learningBlock_"):
            current_block = desc
        elif desc in trial_names:
            if current_block in EARLY_BLOCKS:
                early_trial_times.append(onset)
            elif current_block in LATE_BLOCKS:
                late_trial_times.append(onset)

    return early_trial_times, late_trial_times


def make_events_from_times(raw, times_sec, event_code):
    sfreq = raw.info["sfreq"]
    events = []
    for t in times_sec:
        sample = int(round(t * sfreq))
        events.append([sample, 0, event_code])
    return np.array(events, dtype=int) if len(events) > 0 else np.empty((0, 3), dtype=int)


def compute_epoch_theta(raw, event_times, frontal_chs):
    if len(frontal_chs) == 0:
        raise ValueError("No frontal channels found.")

    if len(event_times) == 0:
        raise ValueError("No event times found for this condition.")

    picks = mne.pick_channels(raw.ch_names, include=frontal_chs)
    events = make_events_from_times(raw, event_times, event_code=1)

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"cond": 1},
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        picks=picks,
        reject_by_annotation=True,
        preload=True,
        verbose=False
    )

    if len(epochs) < MIN_TRIALS_PER_CONDITION:
        raise ValueError(f"Too few usable epochs: {len(epochs)}")

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=THETA_FREQS,
        n_cycles=N_CYCLES,
        use_fft=True,
        return_itc=False,
        average=False,
        verbose=False
    )

    # Percentage baseline normalization
    power.apply_baseline(baseline=BASELINE, mode="percent")

    # Post-stimulus window only
    post_mask = (power.times >= 0.2) & (power.times <= 0.8)

    # Mean theta per epoch
    # shape before mean: epochs x channels x freqs x selected_times
    epoch_means = power.data[:, :, :, post_mask].mean(axis=(1, 2, 3))

    theta_scalar = epoch_means.mean()
    theta_sem = epoch_means.std(ddof=1) / np.sqrt(len(epoch_means)) if len(epoch_means) > 1 else 0.0

    return theta_scalar, theta_sem, len(epochs)


# =========================================================
# MAIN
# =========================================================


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_dir = os.path.join(base_dir, INPUT_FOLDER)
    output_dir = os.path.join(base_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".edf") and file_matches_task_mode(f, TASK_MODE)
    ])

    if len(files) == 0:
        raise RuntimeError("No matching EDF files found.")

    early_vals = []
    late_vals = []
    used_subjects = []
    summary_rows = []
    frontal_reference = FALLBACK_FRONTAL  # now fixed across all subjects

    print("Files to process:")
    for f in files:
        print(" ", f)

    for fname in files:
        subj = extract_subject_code(fname)
        if subj is None:
            print(f"Skipping file with no subject code: {fname}")
            continue

        fpath = os.path.join(input_dir, fname)

        try:
            print(f"\nProcessing {subj} ...")
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)

            raw = crop_to_task(raw)

            frontal_chs = get_frontal_channels(raw, n_channels=N_FRONTAL_CHANNELS)
            if len(frontal_chs) == 0:
                raise ValueError("No frontal channels identified.")

            print("  Frontal channels:", frontal_chs)

            early_trial_times, late_trial_times = assign_trials_to_blocks(raw)

            print(f"  Early trials found: {len(early_trial_times)}")
            print(f"  Late trials found : {len(late_trial_times)}")

            early_theta, early_sem, n_early = compute_epoch_theta(raw, early_trial_times, frontal_chs)
            late_theta, late_sem, n_late = compute_epoch_theta(raw, late_trial_times, frontal_chs)

            print(f"  Early usable epochs: {n_early}")
            print(f"  Late usable epochs : {n_late}")
            print(f"  Early theta: {early_theta:.6e} | SEM: {early_sem:.6e}")
            print(f"  Late theta : {late_theta:.6e} | SEM: {late_sem:.6e}")

            early_vals.append(early_theta)
            late_vals.append(late_theta)
            used_subjects.append(subj)
            summary_rows.append((subj, early_theta, late_theta, early_sem, late_sem, n_early, n_late, fname))

            # subject plot with SEM
            plt.figure(figsize=(5, 5))
            plt.bar(
                [0, 1],
                [early_theta, late_theta],
                yerr=[early_sem, late_sem],
                capsize=6,
                alpha=0.7
            )
            plt.xticks([0, 1], ["Early\n(Block 1-2)", "Late\n(Block 4-5)"])
            plt.ylabel("Mean frontal theta power (% baseline change)")
            plt.title(f"{subj}\nEarly vs Late")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{subj}_early_vs_late_theta.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"  Skipping {subj}: {e}")

    early_vals = np.array(early_vals)
    late_vals = np.array(late_vals)

    if len(used_subjects) < 2:
        raise RuntimeError("Not enough subjects with usable early and late data.")

    # stats
    t_stat, p_ttest = ttest_rel(late_vals, early_vals)
    try:
        w_stat, p_wilcoxon = wilcoxon(late_vals, early_vals)
    except Exception:
        w_stat, p_wilcoxon = np.nan, np.nan

    diff = late_vals - early_vals
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan

    # save summary
    summary_csv = os.path.join(output_dir, "sequence_early_vs_late_theta_summary.csv")
    with open(summary_csv, "w") as f:
        f.write("subject,early_theta,late_theta,early_sem,late_sem,early_epochs,late_epochs,filename\n")
        for row in summary_rows:
            f.write(",".join(map(str, row)) + "\n")

    stats_txt = os.path.join(output_dir, "sequence_early_vs_late_theta_stats.txt")
    with open(stats_txt, "w") as f:
        f.write(f"N subjects: {len(used_subjects)}\n")
        f.write(f"Paired t-test: t = {t_stat:.6f}, p = {p_ttest:.6g}\n")
        f.write(f"Wilcoxon: W = {w_stat}, p = {p_wilcoxon}\n")
        f.write(f"Cohen's d (paired): {cohens_d}\n")
        f.write(f"Mean early theta: {np.mean(early_vals):.6e}\n")
        f.write(f"Mean late theta: {np.mean(late_vals):.6e}\n")

    ch_file = os.path.join(output_dir, "frontal_channels_used.txt")
    with open(ch_file, "w") as f:
        f.write("\n".join(frontal_reference))

    # group plot (SEM across subjects)
    means = [np.mean(early_vals), np.mean(late_vals)]
    sems = [
        np.std(early_vals, ddof=1) / np.sqrt(len(early_vals)),
        np.std(late_vals, ddof=1) / np.sqrt(len(late_vals))
    ]

    plt.figure(figsize=(7, 6))
    x = np.array([0, 1])

    plt.bar(x, means, yerr=sems, capsize=6, alpha=0.7)
    for i in range(len(used_subjects)):
        plt.plot(x, [early_vals[i], late_vals[i]], marker="o", alpha=0.7)

    plt.xticks(x, ["Early\n(Block 1-2)", "Late\n(Block 4-5)"])
    plt.ylabel("Mean frontal theta power (% baseline change)")
    plt.title(f"Sequence Learning: Early vs Late\npaired t-test p = {p_ttest:.4g}")
    plt.tight_layout()

    group_plot = os.path.join(output_dir, "GROUP_sequence_early_vs_late_theta_barplot.png")
    plt.savefig(group_plot, dpi=300)
    plt.close()

    print("\nDone.")
    print(f"Used subjects: {len(used_subjects)}")
    print(f"Paired t-test p = {p_ttest:.6g}")
    print(f"Saved group plot to: {group_plot}")