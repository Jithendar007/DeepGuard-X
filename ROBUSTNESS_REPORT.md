# DeepGuard-X Robustness Report

## 1. Objective

Compare V1 fixed, V2 adaptive, V3 calibrated adaptive, and cross-modal security decisions using recorded attempts.

## 2. Test Protocol

Real captures use the existing six-action protocol. UNIT_TEST and FAILURE_CASE records are excluded from genuine/impostor metrics.

## Genuine results

Recorded attempts: 2

## Impostor results

Recorded attempts: 1

## Degraded capture results

Recorded attempts: 1

## Failure-safe results

Recorded attempts: 1

## V1 vs V2 vs V3 Comparison

| Version | FAR | FRR | TAR | TRR | Accuracy | Mean Runtime |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 32.9025 |
| V2 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 32.9025 |
| V3 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 32.9025 |
| FINAL | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 32.9025 |

## Reliability Behaviour

Reliability summaries are descriptive measurements. No causality or performance improvement is claimed.

## Cross-Modal Consistency Behaviour

Conflict and security-override counts are written to `cross_modal_summary.csv`.

## Limitations

Insufficient labeled samples for statistically meaningful FAR/FRR estimation unless enough genuine and impostor attempts are collected.

## Conclusions

This report describes measured records only and does not claim that V2 or V3 improves performance.
