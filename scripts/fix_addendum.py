import re

with open('docs/wandb_analysis.md', 'r', encoding='utf-8') as f:
    text = f.read()

# remove everything from "## Addendum: W&B Data Structure Notes for Best Model Checkpoints" onwards
text = re.split(r"## Addendum: W&B Data Structure Notes for Best Model Checkpoints", text)[0]

clean_addendum = """
## Addendum: W&B Data Structure Notes for Best Model Checkpoints

When analyzing runs that employ the *4 best model weights* strategy (saved for one-step nrmse/vrmse and rollout nrmse/vrmse), the evaluation metrics on the final test set are stored under specific prefixes. 

### Key Metrics for Best Weights
Instead of evaluating the *final* checkpoint (which may overfit or underfit depending on the epoch), rely on the evaluations of these 4 best weights. The keys in `r.summary` will follow this format:
- `test_best_one_step_nrmse_{dataset}/full_NRMSE_T=all`
- `test_best_one_step_vrmse_{dataset}/full_VRMSE_T=all`
- `test_best_rollout_nrmse_{dataset}/full_NRMSE_T=all`
- `test_best_rollout_vrmse_{dataset}/full_VRMSE_T=all`

### Best Practices for Analysis
1. **Groups over Config Inference**: Always use the wandb Group (`r.group`) when analyzing ablations (e.g. `baseline`, `TB_4`, `PF`) as it is far more reliable and easier to read than reverse-engineering the config dictionary.
2. **Apples-to-Apples Comparisons**: Compare the same evaluation metric across different 'best' weights. For example, you should compare how the 'best one-step NRMSE' model performs on rollout NRMSE vs how the 'best rollout NRMSE' model performs on rollout NRMSE.
3. **Variance in Best Models**: In some architectures (like FNO), the checkpoint with the best *validation* rollout NRMSE performs worse on the *test* set than the checkpoint with the best *validation* one-step NRMSE. This highlights the importance of having all 4 best weights stored.
4. **Hardware and AMP Info**: Hardware specs (GPU) are sometimes difficult to extract solely from `r.config` as they might be stored in the W&B run metadata. For accurate runtime comparisons, check whether `r.config.get('trainer', {}).get('amp')` or `r.config.get('amp')` is true, as mixed-precision drastically affects runtime.
"""

with open('docs/wandb_analysis.md', 'w', encoding='utf-8') as f:
    f.write(text + clean_addendum)

print("Addendum fixed!")
