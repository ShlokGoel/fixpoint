# Technical write-up (template)

Fill this in from real runs only. Every number should come from a `summary.md` or a
`compare` output, with the run name next to it.

## 1. Setup
- Model ID and served model string, plus thinking mode and effort level (from `run_meta.json`)
- Config fingerprint, prompt hashes, agent git commit
- Dataset and split, subset size and seed, and the number of repeats
- Budgets per task (steps, dollars, wall-clock)

## 2. Headline results
- Resolved % with a 95% CI, for the 30-task subset and the full 300
- Cost: total, mean, median and p90 per task, and cost per resolved task
- Run-to-run spread across repeats

## 3. Failure taxonomy
- The primary-category table, which sums to N
- Two or three representative trajectories per major category: what went wrong, at which step
- Which secondary tags occur alongside failures (for example, does `context_reset` predict failure?)

## 4. Reward hacking
- Every flagged instance, with its resolved status and the evidence lines
- Whether any *resolved* task carried a high-severity flag, and how that changes the headline number

## 5. Ablations
- No localization vs. full scaffold: paired delta, bootstrap CI, McNemar p
- No self-verification vs. full scaffold: the same
- How the failure mix shifts, not just the resolve rate

## 6. What I would try next
- Grounded in the taxonomy: the biggest failure bucket gets the next experiment
