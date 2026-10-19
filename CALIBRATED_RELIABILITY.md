# Calibrated Reliability-Aware Adaptive Fusion

DeepGuard-X now supports three comparable decision views:

- **Version 1:** fixed score-level fusion.
- **Version 2:** existing reliability-aware adaptive fusion.
- **Version 3:** calibrated reliability-aware adaptive score-level fusion with cross-modal security consistency analysis.

Version 3 is a deterministic, explainable prototype. It is not machine-learned calibration and no performance or novelty claim is made.

## Why Fixed Fusion Is Limited

Version 1 assigns fixed weights even when one modality has few valid frames, inconsistent predictions, or poor capture quality. A fixed score can therefore hide evidence-quality differences between attempts.

## Why Raw Heuristic Reliability Is Limited

Version 2 uses deterministic reliability estimates, but its components are compact heuristics. Version 3 makes the measurable components, weights, thresholds, and calibration transform explicit and configurable so experiments can reproduce them.

## Available Measurements

### Face ID

- Selected Face ID similarity score.
- Number of valid selected-user recognition samples.
- Standard deviation of selected-user frame scores.
- Shared sharpness/illumination quality from captured frames.

### Behaviour

- Valid behaviour rows divided by the target of 100 rows.
- Fraction of rows containing all ten finite behaviour features.
- Temporal consistency from the standard deviation of the five delta features.
- Existing behaviour score.

### PAD

- Processed PAD frames divided by the configured expected heavy-model frame count of 20.
- Distance of the PAD real score from the ambiguous midpoint 0.50.
- Standard deviation of per-frame PAD real predictions.
- Shared sharpness/illumination quality.

### Deepfake

- Processed Deepfake frames divided by the configured expected heavy-model frame count of 20.
- Distance of the Deepfake real score from the ambiguous midpoint 0.50.
- Standard deviation of per-frame Deepfake real predictions.
- Shared sharpness/illumination quality.

No raw images, biometric templates, or model weights are logged.

## Normalization and Formula

Every quality component is bounded to `[0, 1]`. Consistency is deterministic:

```text
consistency = exp(-standard_deviation / consistency_scale)
```

Score confidence is:

```text
score_confidence = clip(2 * abs(score - 0.50), 0, 1)
```

For each modality:

```text
Q = sum(alpha_i * q_i) / sum(alpha_i)
R = clip(Q ** calibration_power, 0, 1)
```

`alpha_i`, scales, targets, and calibration power are centralized in `ReliabilityConfig` in `calibrated_reliability.py`.

The default `calibration_power` is `1.0`, so the initial prototype calibration is transparent and deterministic. No calibration dataset or labels are currently used.

## Version 3 Adaptive Weights

Version 3 uses the unchanged baseline weights:

```text
Face       = 0.30
Behaviour  = 0.25
PAD        = 0.25
Deepfake   = 0.20
```

```text
raw_weight_i = baseline_weight_i * calibrated_reliability_i
adaptive_weight_i = raw_weight_i / sum(raw_weight_i)
```

The resulting calibrated adaptive score is:

```text
V3 = sum(adaptive_weight_i * model_score_i)
```

If required evidence is unavailable, Face ID is not a match, a score is invalid, or weights cannot be formed, Version 3 fails closed with `REJECT`.

## Security Constraints

- PAD score below `0.50` rejects.
- Invalid or missing required evidence rejects.
- NaN and infinity reject.
- Invalid Face ID rejects.
- Cross-modal `HIGH_RISK` or `CRITICAL` security override rejects.
- Behaviour-only contradiction remains a warning and does not independently reject.
- The existing cross-modal contradiction layer is evaluated after fusion.

## Cross-Modal Interaction

Cross-modal analysis uses the four model scores and computes pairwise agreement:

```text
agreement(a, b) = 1 - abs(a - b)
```

The six pairwise agreements are averaged. Contradictions such as Face ID plus low PAD or Face ID plus low Deepfake-Real produce security overrides according to the existing rule set.

## Experiment Protocol

Use the same person, camera, and controlled six-action behaviour protocol for:

1. Normal lighting.
2. Mildly reduced lighting.
3. Reproducible mild blur or lower quality.
4. Slight head pose.
5. Full normal evidence.
6. Simulated missing/invalid modality inputs.
7. Controlled PAD-failure test inputs only.
8. Deterministic contradiction unit tests.

Run `evaluate_fusion.py` after attempts to create `fusion_evaluation.jsonl` and `fusion_evaluation.csv`.

## Metrics and Limitations

The repository currently does not contain enough labeled genuine/impostor attempts for statistically meaningful FAR/FRR estimation. Do not fabricate FAR, FRR, accuracy, or spoof-rejection metrics. The evaluation harness reports descriptive per-attempt values only until labeled data is collected with separate calibration and test partitions.

The reliability parameters are configurable prototype parameters, not statistically learned calibration. The method has not been demonstrated to outperform Version 1 or Version 2.

## Contribution Scope

The intended research description is:

> calibrated reliability-aware adaptive score-level fusion with cross-modal security consistency analysis

This is a proposed research contribution. No claim is made about legal novelty, patentability, state-of-the-art performance, or superiority without prior-art review and controlled experiments.
