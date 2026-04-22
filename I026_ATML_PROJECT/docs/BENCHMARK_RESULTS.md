# Benchmark Results

## Latest validated run

- Command:
  - `python kaggle_eval/run_kaggle_validation.py --ae-epochs 80 --gan-epochs 160 --batch-size 512`
- Device:
  - CUDA GPU (RTX 3060)
- Cases:
  - 7
- Overall similarity average:
  - `0.8939`
- Pass criterion:
  - all cases `> 0.8` overall similarity
- Result:
  - passed

## Per-case overall similarity

| Case | Score |
|---|---:|
| telco_churn | 0.9266 |
| stroke_prediction | 0.8613 |
| adult_income | 0.9055 |
| heart_failure | 0.9059 |
| credit_card_fraud | 0.8910 |
| pima_diabetes | 0.8835 |
| diabetes_general | 0.8835 |

## Notes

- Full raw metrics are generated under `kaggle_eval/output/` during execution.
- That output directory is intentionally git-ignored to avoid committing large synthetic artifacts.

