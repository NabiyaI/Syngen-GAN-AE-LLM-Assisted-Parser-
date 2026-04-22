# Known Limitations

1. **Supported domains only**: This tool is optimized for Healthcare, Finance, and Ecommerce domains only.
   - Prompts for unsupported domains (telecom, education, saas, etc.) will be rejected.
   - Default fallback is Healthcare domain.

2. Prompt coverage is broad for supported domains but not universal.
   - Free-form prompts can still fail if constraints are ambiguous or contradictory.

3. LLM rewrite is non-deterministic without fixed model/version behavior.
   - Use `rules` mode for strict reproducibility.

4. Domain quality depends on checkpoint quality.
   - Weak or undertrained checkpoints reduce realism.

5. The generator can enforce hard filters post-sampling.
   - This can improve constraints but slightly distort joint distributions.

6. Kaggle evaluation is similarity-based, not privacy-safe synthesis certification.
   - Additional privacy and utility checks are needed for production use.

7. GUI is Windows-first for one-click CSV opening (`os.startfile`).
   - Linux/Mac users may need manual CSV opening.

