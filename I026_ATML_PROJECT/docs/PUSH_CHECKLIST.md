# Push Checklist

1. Security
   - Confirm `.env` is not tracked.
   - Rotate any previously shared API keys.
2. Repo hygiene
   - Ensure `checkpoints/` and `output/` large artifacts are ignored.
   - Keep only config/registry and lightweight reports in Git.
3. Validation
   - Run `pytest -q`.
   - Run one CLI smoke test and one GUI generation.
4. Docs
   - Update README metrics table if rerun benchmarks.
   - Ensure setup instructions match actual environment.
5. Commit quality
   - Use clear commit messages per feature area.
   - Do not commit local machine paths or secrets.

