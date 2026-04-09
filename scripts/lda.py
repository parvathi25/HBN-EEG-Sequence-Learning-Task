import os
import re
import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================

INPUT_FOLDER = "sequence_input"
OUTPUT_FOLDER = "lda_sequence_output"

TMIN = -0.5
TMAX = 1.5
BASELINE = (-0.5, 0.0)

THETA_FREQS = np.arange(4, 8)   # 4, 5, 6, 7 Hz
N_CYCLES = THETA_FREQS / 2.0

FALLBACK_FRONTAL = ["E3", "E4", "E9", "E10", "E11", "E15", "E16", "E18", "E19", "E22", "E23"]

MIN_TRIALS_PER_CLASS = 5

EARLY_BLOCKS = {"learningBlock_1", "learningBlock_2"}
LATE_BLOCKS = {"learningBlock_4", "learningBlock_5"}

POST_WINDOW = (0.2, 0.8)   # seconds

# =========================
# HELPERS
# =========================

def extract_subject_code(filename):
    m = re.search(r"(sub-[A-Za-z0-9]+)", filename)
    return m.group(1) if m else None


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
    block_trials = {
        "learningBlock_1": [],
        "learningBlock_2": [],
        "learningBlock_3": [],
        "learningBlock_4": [],
        "learningBlock_5": []
    }

    for onset, desc in anns:
        if desc in block_trials:
            current_block = desc
        elif desc in trial_names:
            if current_block is not None:
                block_trials[current_block].append(onset)

    return block_trials


def make_events_from_times(raw, times_sec, event_code):
    sfreq = raw.info["sfreq"]
    events = []
    for t in times_sec:
        sample = int(round(t * sfreq))
        events.append([sample, 0, event_code])
    return np.array(events, dtype=int) if len(events) > 0 else np.empty((0, 3), dtype=int)


def extract_theta_features(raw, times, picks):
    """
    Returns one feature vector per epoch.
    Features = mean theta power in each frontal channel.
    """
    if len(times) < MIN_TRIALS_PER_CLASS:
        return None

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

    if len(epochs) < MIN_TRIALS_PER_CLASS:
        return None

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=THETA_FREQS,
        n_cycles=N_CYCLES,
        return_itc=False,
        average=False,
        verbose=False
    )

    # percent baseline normalization
    power.apply_baseline(BASELINE, mode="percent")

    mask = (power.times >= POST_WINDOW[0]) & (power.times <= POST_WINDOW[1])
    data = power.data[..., mask]   # epochs x ch x freq x time

    # mean over freq and time, keep channels as features
    features = data.mean(axis=(2, 3))   # epochs x channels

    return features


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_dir = os.path.join(base, INPUT_FOLDER)
    output_dir = os.path.join(base, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".edf")])

    subject_accs = []
    subject_names = []

    summary_rows = []

    for f in files:
        subj = extract_subject_code(f)
        if not subj:
            continue

        print(f"\nProcessing {subj}")

        raw = mne.io.read_raw_edf(os.path.join(input_dir, f), preload=True, verbose=False)
        raw = crop_to_task(raw)

        frontal = get_frontal_channels(raw)
        if len(frontal) == 0:
            print("  No frontal channels found, skipping.")
            continue

        print("  Frontal channels:", frontal)

        block_trials = assign_trials_per_block(raw)

        early = block_trials["learningBlock_1"] + block_trials["learningBlock_2"]
        late = block_trials["learningBlock_4"] + block_trials["learningBlock_5"]

        print("  Early trials:", len(early))
        print("  Late trials :", len(late))

        X_early = extract_theta_features(raw, early, frontal)
        X_late = extract_theta_features(raw, late, frontal)

        if X_early is None or X_late is None:
            print("  Skipping due to too few usable trials.")
            continue

        y_early = np.zeros(len(X_early), dtype=int)   # 0 = early
        y_late = np.ones(len(X_late), dtype=int)      # 1 = late

        X = np.vstack([X_early, X_late])
        y = np.concatenate([y_early, y_late])

        print("  Feature matrix shape:", X.shape)

        # LDA with standardization
        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis()
        )

        # number of folds limited by smallest class
        n_early = len(X_early)
        n_late = len(X_late)
        n_splits = min(5, n_early, n_late)

        if n_splits < 2:
            print("  Not enough trials for cross-validation, skipping.")
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        mean_acc = scores.mean()
        std_acc = scores.std(ddof=1) if len(scores) > 1 else 0.0

        print(f"  LDA accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

        subject_names.append(subj)
        subject_accs.append(mean_acc)

        summary_rows.append((subj, mean_acc, std_acc, len(X_early), len(X_late), X.shape[1]))

    subject_accs = np.array(subject_accs)

    if len(subject_accs) == 0:
        raise RuntimeError("No subjects had usable data for LDA.")

    # Save summary CSV
    summary_path = os.path.join(output_dir, "lda_subject_summary.csv")
    with open(summary_path, "w") as f:
        f.write("subject,mean_accuracy,std_accuracy,n_early_trials,n_late_trials,n_features\n")
        for row in summary_rows:
            f.write(",".join(map(str, row)) + "\n")

    # Save stats text
    stats_path = os.path.join(output_dir, "lda_group_summary.txt")
    with open(stats_path, "w") as f:
        f.write(f"N subjects: {len(subject_accs)}\n")
        f.write(f"Mean subject accuracy: {subject_accs.mean():.6f}\n")
        f.write(f"SD subject accuracy: {subject_accs.std(ddof=1) if len(subject_accs)>1 else 0.0:.6f}\n")
        f.write("Chance level: 0.5\n")

    # Plot subject accuracies
    plt.figure(figsize=(10, 6))
    x = np.arange(len(subject_names))
    plt.bar(x, subject_accs, alpha=0.7)
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Chance")
    plt.xticks(x, subject_names, rotation=90)
    plt.ylabel("LDA accuracy")
    plt.title("Early vs Late Classification Accuracy by Subject")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lda_subject_accuracies.png"), dpi=300)
    plt.close()

    print("\nDone.")
    print(f"Subjects used: {len(subject_accs)}")
    print(f"Mean accuracy: {subject_accs.mean():.3f}")
    print(f"Saved outputs to: {output_dir}")