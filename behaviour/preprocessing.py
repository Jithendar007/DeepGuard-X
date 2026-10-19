import numpy as np


FEATURE_COLUMNS = [
    "ear",
    "mar",
    "yaw",
    "pitch",
    "roll",
    "d_ear",
    "d_mar",
    "d_yaw",
    "d_pitch",
    "d_roll"
]
TARGET_FRAMES = 120


def preprocess_sequence(raw_sequence):
    """Unwrap pose, recalculate pose deltas, and resample to 120 frames."""
    sequence = np.asarray(raw_sequence, dtype=np.float32).copy()

    if sequence.ndim != 2 or sequence.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected a sequence shaped (frames, {len(FEATURE_COLUMNS)})."
        )
    if len(sequence) < 2:
        raise ValueError("At least two frames are required.")

    for index in (2, 3, 4):
        radians = np.deg2rad(sequence[:, index])
        sequence[:, index] = np.rad2deg(np.unwrap(radians))

    sequence[:, 7:10] = 0.0
    sequence[1:, 7] = np.abs(np.diff(sequence[:, 2]))
    sequence[1:, 8] = np.abs(np.diff(sequence[:, 3]))
    sequence[1:, 9] = np.abs(np.diff(sequence[:, 4]))

    if len(sequence) == TARGET_FRAMES:
        return sequence

    old_indices = np.linspace(0, len(sequence) - 1, len(sequence))
    new_indices = np.linspace(0, len(sequence) - 1, TARGET_FRAMES)
    resampled = np.zeros(
        (TARGET_FRAMES, len(FEATURE_COLUMNS)),
        dtype=np.float32
    )
    for feature_index in range(len(FEATURE_COLUMNS)):
        resampled[:, feature_index] = np.interp(
            new_indices,
            old_indices,
            sequence[:, feature_index]
        )

    return resampled


def normalize_sequence(sequence, mean, std):
    """Apply the saved feature-wise normalization parameters."""
    normalized = (np.asarray(sequence) - mean) / std
    return normalized.astype(np.float32)
