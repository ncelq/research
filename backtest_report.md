# QQQ Direction Backtest Report

- Range: 2020-01-01..2026-08-30
- Timepoints (N): 50 (target: N+1)
- Workers: 20
- Model: client default (same path as predict_qqq_direction.py)
- Generated: 2026-09-29T10:09:45Z
- Predicted days: 1523
- Warmup-skipped days: 0
- Error-skipped days: 150

## Overall accuracy

- Correct: 688/1523
- Accuracy: 0.4517
- Majority-class baseline: 0.5542

## Confusion matrix (UP = positive)

|  | Actual UP | Actual DOWN |
|---|---|---|
| Predicted UP | 58 | 49 |
| Predicted DOWN | 786 | 630 |

- Precision: 0.5421
- Recall: 0.0687

## Accuracy by year

| Year | Correct | Total | Accuracy |
|---|---|---|---|
| 2020 | 85 | 203 | 0.4187 |
| 2021 | 106 | 252 | 0.4206 |
| 2022 | 138 | 251 | 0.5498 |
| 2023 | 70 | 158 | 0.4430 |
| 2024 | 103 | 244 | 0.4221 |
| 2025 | 107 | 250 | 0.4280 |
| 2026 | 79 | 165 | 0.4788 |

## Accuracy by confidence

| Bucket | Correct | Total | Accuracy |
|---|---|---|---|
| conf<0.6 | 680 | 1507 | 0.4512 |
| conf0.6-0.8 | 8 | 16 | 0.5000 |
| conf>=0.8 | 0 | 0 | 0.0000 |
