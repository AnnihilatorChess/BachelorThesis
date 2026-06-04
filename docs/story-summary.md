# Appendix A — Story Summary

*Abbreviations. OS: one-step training. BPTT: backpropagation through time. PF: pushforward; TB: temporal bundling; TB+PF: their combination. Datasets: TRL and AM are chaotic problems from The Well; SWE and BUR are simpler problems from PDEBench. Architectures: FNO (global/spectral), UNet (local/convolutional), CNO (continuous convolutional operator). Metrics: VRMSE on The Well data, NRMSE on PDEBench data, lower is better. Scales: the small (BPTT-capacity) scale is trained for FNO and UNet on SWE and BUR only; the large scale is trained for all three architectures on all four datasets.*

---

**What is the central question?**

How effectively do lightweight stabilization techniques (noise injection, pushforward, temporal bundling, and TB+PF) improve the autoregressive rollout accuracy of neural PDE surrogates over plain one-step training, and how do they compare to BPTT in accuracy and memory cost? And because they keep the memory footprint low, do they let us train larger models that outperform the small models BPTT is restricted to?

**Why is this question important?**

For transient PDEs, a surrogate is only useful if its autoregressive rollout stays accurate over long horizons, since rollout, not one-step accuracy, is the real measure of success. BPTT trains for this directly but stores the full unrolled computation graph, so its memory cost restricts it to small models. One-step training is cheap but causes distribution shift: the model is trained on ground-truth inputs, yet at inference must consume its own imperfect predictions and diverges.

**What evidence/data (variables) are needed to answer this question?**

For each combination of architecture, dataset, and training method, we need the rollout error (overall and per time window), the one-step error, the peak VRAM, and the wall-clock time. A generalization claim requires several architectures (global, local, operator-based) and several PDEs with distinct physics. The capacity claim requires two model scales on the same dataset: a small scale that fits BPTT in memory, and a large scale that only the lightweight methods can afford. We realise this two-scale design only on the PDEBench datasets (SWE, BUR), where a published BPTT baseline exists; on the chaotic Well datasets (TRL, AM) we currently train only at the large scale.

**What methods are used to get this evidence/data?**

We train FNO, UNet, and CNO on TRL, AM, SWE, and BUR with the full ablation stack (baseline, Noise, PF, TB, TB+PF), repeating each run over several seeds for mean and standard deviation. On the two PDEBench datasets we additionally run a step-count-exact BPTT trainer at small capacity as the reference, for FNO and UNet only — CNO has no PDEBench baseline to reproduce, so it is trained only at the large scale. Each configuration uses a per-case learning-rate search and four tracked checkpoints, so we can separate best one-step weights from best rollout weights.

**What analyses must be applied to the data to answer the central question?**

We compare each technique's rollout error against the one-step baseline, the published paper baselines, and BPTT at matched small capacity, alongside the corresponding VRAM and runtime. We then run the capacity-stability comparison: large stabilized models against small BPTT-constrained models. Secondary analyses cover checkpoint selection, context-window length (SWE at 4 vs 10 input steps), and the implicit regularization from mixed-precision noise on AM.

**What evidence/data (values for the variables) were obtained?**

On the chaotic datasets, TB+PF was the strongest stabilizer, cutting rollout VRMSE over the one-step baseline by about 55% for FNO, 38% for UNet, and 30% for CNO on TRL, and 41% for UNet on AM. On the simpler PDEBench datasets, large TB+PF models reached the best rollout NRMSE overall and beat small-model BPTT, for example FNO at 0.0012 versus BPTT at 0.0022 on SWE. At matched small capacity, however, BPTT remained best for FNO (0.0022 on SWE, 0.047 on BUR), so stabilization only overtakes BPTT once its saved memory buys a larger model.

*Note: these are the current best-rollout numbers from the results docs. AM is still running, so its final values must be re-checked before submission.*

**What were the results of the analyses?**

Stabilization consistently improves autoregressive rollout over one-step training, with TB+PF strongest and noise injection weakest, at minor extra cost (PF adds roughly 40 to 50% wall-clock, TB almost none). The techniques trade a small loss in one-step accuracy for a large gain in rollout stability, confirming they act as rollout-oriented regularizers. Their low VRAM footprint allows roughly 18M-parameter models that match or exceed the small models BPTT is limited to, though BPTT stays competitive for FNO at matched small capacity.

**How did the analyses answer the central question?**

The comparisons show that lightweight stabilization, mainly TB+PF, reduces distribution shift and improves rollout across all three architectures and all four PDEs at a fraction of BPTT's memory cost. On the PDEBench datasets (SWE, BUR), where we hold both a small and a large scale for FNO and UNet, the capacity-stability analysis shows that the freed memory, reinvested in larger models, lets them surpass the small models BPTT is restricted to. Together this answers the central question: stabilization improves rollout everywhere we tested, and — on the datasets where we can compare scales — it is an economical alternative to BPTT precisely because it unlocks model capacity. Whether this capacity advantage extends to the chaotic Well datasets is not yet established (see limitations).

**What does this answer tell us about the broader field?**

For transient-PDE surrogates, the relevant trade-off is the joint optimization of rollout accuracy, model capacity, and memory budget. On the simpler PDEBench problems we find a Pareto frontier on which lightweight stabilization, reinvested in a larger model, beats the small models BPTT is confined to; we did not test this frontier on the chaotic Well problems, where intrinsic error growth may limit the benefit of added capacity. Practitioners building autoregressive surrogates should default to temporal bundling with pushforward over full-trajectory BPTT for its stabilization benefit and low memory cost. They should also select checkpoints and context windows by rollout quality, with the optimal regularizer mildly dependent on whether the operator is global or local.

**Did the work answer the question satisfactorily? Why (not)?** *(Seminar-relevant; optional for the thesis.)*

Largely yes for the stabilization claim, which rests on a broad ablation across three architectures and four PDEs with multi-seed statistics and direct comparison to the paper baselines. The capacity-vs-BPTT claim is narrower: the two-scale comparison and the step-count-exact BPTT reimplementation exist only for FNO and UNet on the two PDEBench datasets (SWE, BUR) — CNO has no PDEBench baseline, and the chaotic Well datasets (TRL, AM) are trained only at the large scale. So the capacity-stability result is demonstrated on the simpler PDEs but not yet on the chaotic ones, where added capacity may matter less; running the small scale on TRL (after the AM runs finish) is the planned step toward closing this gap. Further caveats: AM is incomplete and covers only UNet, and the mixed-precision finding was opportunistic rather than a controlled ablation. These bound the generalization claim but do not change the core stabilization result, which is consistent across every configuration tested.
