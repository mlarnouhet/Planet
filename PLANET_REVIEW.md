Review date: 2026-09-29. Local revision: `3d63db502437db0ddad1e878a7f4eed631133b6f`.

Your current implementation has the core PlaNet algorithm right, and the major image-loss, training-mode, action-bounding, and initial-state problems in the previous review have been fixed. It is still a PlaNet variant rather than a faithful reproduction: the environment setup, CEM spread, reward objective, architecture, optimizer, replay, and evaluation differ. There are also reproducible runtime and data-handling bugs to fix before testing other tasks or launching long runs.

I read the supplied [PLANET.pdf](/home/mlarn/planet/PLANET.pdf), including its appendices, the current Python source, debug replay configuration, and training log. I compared them with the authors' [official implementation](https://github.com/google-research/planet/tree/c04226b6db136f5269625378cd6a0aa875a92842), pinned to `c04226b6db136f5269625378cd6a0aa875a92842`, and ran component checks against your installed dependencies. Your PDF is arXiv v5, June 4, 2019. The released code includes changes since the paper; the two references are distinguished below. This replaces the previous review of revision `b7ec852`.

The current code already gets these details right:

| Component | Assessment at the reviewed revision |
|---|---|
| RSSM structure | Deterministic recurrence, stochastic prior, observation-conditioned posterior, reparameterized samples, and state-conditioned image/reward heads follow the paper's factorization. |
| Image objective | [utils.py:109](/home/mlarn/planet/utils.py:109) now uses half the sum of pixel squared errors, averaged over batch/time. This matches the unit-variance Gaussian objective up to a constant. The old 6,144× image/KL scaling discrepancy is gone. |
| KL | The expression is `KL(posterior || prior)`, summing latent coordinates before applying 3 free nats. My numerical check agreed with PyTorch distributions within `3.1e-5` in float32. |
| Modes and initialization | [main.py:89](/home/mlarn/planet/main.py:89) restores training mode each fitting phase. Collection uses evaluation mode and no gradients, and now performs the same initial zero-state/zero-action GRU transition as training. |
| CEM bounds and selection | Proposals and noisy executed actions are clipped to `[-1,1]`; recurrence and replay use the executed action. GPU `topk` now uses the requested `K`. |
| Rendering | Training renders directly at 64×64 with camera 0 after action repetition. The old wrapper overhead and camera mismatch are gone. |
| MPC | Planning uses prior rollouts and reward means without decoding images, executes the first fitted action mean, and replans after the next observation. |

The final RSSM agent does **not** require latent overshooting or a fixed global prior: see PDF §4 and Appendices A/D; released defaults disable overshooting. One stochastic trajectory per CEM candidate and restarting CEM at zero mean/unit variance are deliberate. An actor, critic, terminal value function, and discount factor are not missing requirements of this finite-horizon baseline. See the [paper](https://arxiv.org/abs/1811.04551) and [released planner](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/planning.py).

The remaining correctness and experiment-validity findings are ordered by practical impact.

1. **High when changing tasks: reward conversion crashes on four benchmark tasks.** Both collection paths call `time_step.reward.astype(np.float32)` at [main.py:79](/home/mlarn/planet/main.py:79) and [main.py:164](/home/mlarn/planet/main.py:164). Actual steps with your installed dm_control returned Python `float` rewards for reacher/easy, cheetah/run, finger/spin, and ball_in_cup/catch. Each raises `AttributeError: 'float' object has no attribute 'astype'`. Cartpole and walker returned NumPy scalars and passed. Use `float(time_step.reward)` with the existing `None` check, then explicitly construct float32 replay tensors.

2. **High for task portability: action dimensions are independent of the environment.** Models are constructed before reading `env.action_spec()`, and [main.py:219](/home/mlarn/planet/main.py:219) defaults to one action coordinate. Measured dimensions are cartpole 1; reacher/finger/cup 2; cheetah/walker 6. Changing only the task can mismatch replay/model shapes or supply invalid actions. Construct the environment first and derive dimensions from its spec. Repeat 8 is also cartpole-specific: use 4 for reacher/cheetah/cup and 2 for finger/walker for paper comparisons. [Official task definitions](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/tasks.py).

3. **High for benchmark fidelity: reward visualization changes observations.** [main.py:35](/home/mlarn/planet/main.py:35) passes `visualize_reward=True`. dm_control changes object/material colors as a function of reward, creating an extra visual cue. The official environment leaves it disabled. Set it to false and regenerate seed data for a comparable experiment. Rendering size and camera are now correct, but this difference remains. [dm_control task behavior](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/base.py), [official environment setup](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/tasks.py).

4. **High for configuration reliability: debug replay replaces requested settings.** `--debug` defaults to true, and [main.py:60](/home/mlarn/planet/main.py:60) replaces the configured dataset with a pickled object. The supplied object has batch size 50, sequence length 50, action dimension 1, repeat 8, and five episodes of 125 frames, without task identity. Changing CLI batch size can therefore disagree with the actual batch; changing tasks can mix incompatible experience. Make normal collection the default and validate imported replay metadata. Currently, `--debug 0` selects fresh collection.

   Separately, the actual parser treats both `--resume False` and `--setup_wandb False` as **true**, because they use `type=bool`. Use `BooleanOptionalAction` or explicit flag actions, including for debug mode.

5. **Medium: CEM spread matches the PDF pseudocode but differs from the release.** [cem.py:39](/home/mlarn/planet/cem.py:39) computes absolute deviations divided by `K-1`. Appendix B prints this, so it is not a transcription error. However, it is not the Gaussian maximum-likelihood standard deviation. Identical elites give zero spread; `K=1` divides by zero and produced a nonfinite action in my probe. For released-code parity:

   ```python
   mu_q = top_k_actions.mean(dim=0)
   sigma_q = (top_k_actions.var(dim=0, correction=0) + 1e-6).sqrt()
   ```

   Validate `1 <= K <= n_candidate_samples` with this replacement. The current estimator requires `K >= 2`. [Official variance update](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/planning.py#L49).

6. **Medium for objective fidelity: reward loss has twice the released coefficient.** [utils.py:106](/home/mlarn/planet/utils.py:106) uses MSE multiplied by `reward_scale=10`. A unit-variance Gaussian contributes `0.5 * squared_error`; the release scales that likelihood by 10. Your per-valid-target reward contribution is twice the reference relative to image loss and KL. Use `reward_loss = 0.5 * mse_loss(...)` for parity. Keeping 10× MSE is a valid weighted objective, but should be documented and ablated. [Reward distribution](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/basic.py), [objective reduction](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/training/utility.py#L192).

7. **Medium: reward indexing is correct internally, but episode endpoints are lost.** Your row stores `(o_t, a_t, reward_received_after_a_t)`. The posterior for `o_{t+1}` must predict the reward stored beside `a_t`, so `reward[k-1]` in [utils.py:95](/home/mlarn/planet/utils.py:95) is correct. A synthetic target check confirmed this. Changing to `reward[k]` alone would introduce a bug.

   Each 50-row chunk provides 49 reward targets, divided by 50. More seriously, the final action/reward of every episode is never trained because its resulting observation is not saved. [main.py:166](/home/mlarn/planet/main.py:166) stores the previous image and later renders the terminal image without saving it. Store `N+1` observations for `N` transitions, or adopt the official initial dummy-action/reward row followed by resulting-observation rows. Define masks and denominators explicitly. [Official replay convention](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/wrappers.py#L429).

8. **Medium outside the default episode length: collection can cross resets.** Neither repeat loop checks `time_step.last()`. A real cartpole probe confirmed that the next step after termination resets the environment. Early termination, longer `T`, or repeat overruns can silently combine episodes in one trajectory. Default cartpole happens to end exactly at 1,000 base steps. Stop both loops at termination and respect the remaining base-step budget.

   [dataset.py:20](/home/mlarn/planet/dataset.py:20) uses configured `T/repeat`, rather than actual episode length, to sample windows. A two-frame episode requested as a three-frame chunk returned an undersized batch in my probe. Sample from actual lengths; reject short episodes or pad and mask them.

9. **Medium for older checkpoints: replay decoding assumes the new format.** [dataset.py:37](/home/mlarn/planet/dataset.py:37) interprets all observations as integer codes 0–31. New-format save/load preserved quantization bins in my test. Passing normalized float replay from the earlier format instead compressed its values to approximately `[-0.512, -0.454]`. Add a checkpoint version and observation-encoding field; migrate supported old float replay or reject it. Raw uint8 pixels 0–255 are another distinct representation. Architecture changes can independently make older model weights incompatible.

10. **Medium for recovery: configuration and random states are not checkpointed.** Global NumPy seeding does not seed dm_control's separate `RandomState`; pass `task_kwargs={"random": args.seed}`. Save configuration, counters, Python/NumPy/Torch RNG states, and task RNG state for repeatable episode-boundary continuation. Default periodic saves end at step 950 even though training continues through 999; save on completion too. [main.py:175](/home/mlarn/planet/main.py:175).

    `setup_dirs()` recursively deletes existing run directories when not resuming; prefer a fresh run ID or collision error. HF upload is unconditional at save time and immediately awaited, so authentication/network failures can stop a local training run. Make upload optional and handle failures after local persistence. [utils.py:24](/home/mlarn/planet/utils.py:24), [main.py:191](/home/mlarn/planet/main.py:191). I did not execute the training entry point.

The architectural and training differences below are not all bugs. Several are reasonable experiments, but need separate ablations to explain differences in results.

| Component | Your current implementation | Paper / official release |
|---|---|---|
| State sizes | Deterministic 200, stochastic 30 | Same |
| CNN | Five padded 3×3 convs; channels 16/32/64/128/128; first stride 1; BatchNorm; projection to 30 | Release: four valid 4×4 stride-2 convs, channels 32/64/128/256; 1,024 features; no BatchNorm |
| Posterior | 30 image + 200 hidden features; hidden width 230 | Release: 1,024 image + 200 hidden features; hidden width 200 |
| GRU input | Direct state/action concatenation | Release: dense/ReLU projection to 200 before GRU |
| Prior | Two hidden layers of width 200 | Release: one hidden layer of width 200 |
| Reward head | Two hidden layers of width 230 | PDF describes two 200-unit layers; release uses three 300-unit layers |
| Decoder | Dense to 128×4×4; four 3×3 transpose convs with BatchNorm; final conv | Release: dense to 1,024×1×1; transpose-conv kernels 5/5/6/6; no BatchNorm |
| Latent std floor | 0.01 | Release: 0.1 |
| Adam | lr 1e-3, epsilon 1e-8, weight decay 1e-4 | PDF/release: lr 1e-3, epsilon 1e-4; no weight decay in released optimizer |
| Gradient clipping | Each of five modules capped separately at norm 1,000 | Release: one global norm capped at 1,000 |
| Free nats | `max(KL,3)` | Release: `max(KL-3,0)`; additive constant difference, equivalent gradients away from threshold |
| CEM budget | Horizon 12, iterations 10, candidates 1,000, elites 100 | Same defaults; spread estimator differs |
| Fitting cadence | 100 updates then one episode | Same as PDF; release's `collect_every=5000` counts batch-size increments, also 100 updates at batch size 50 |
| Replay | Uniform episode/window sampling; float images in RAM; noise fixed at collection/load | Release: recent/cached episode mixture; uint8 images; fresh dequantization during batch preprocessing |
| Evaluation | Recent noisy training returns only | Release: separate test episodes without action exploration noise, plus prediction diagnostics |

Architecture sources: [RSSM](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/models/rssm.py), [image networks](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/conv_ha.py), [head networks](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/basic.py). Training sources: [configuration](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py), [global clipping](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/custom_optimizer.py), [replay loader](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/numpy_episodes.py).

The 30-dimensional image embedding imposes an extra bottleneck before the sampled state. BatchNorm introduces dependence on batch composition and running-statistics behavior during acting. These are valid variants, but an official-architecture baseline would clarify their impact. PyTorch GRUCell also does not guarantee identical gate equations or initialization to TensorFlow GRUBlockCell; exact numerical equivalence needs an explicit mapping.

Your newest [training log](/home/mlarn/planet/logs/train_1.log:16) reports image loss 1,623.52, reward loss 3.452, clipped KL 3.0, zero prior-head gradient norm, and mean preclip decoder gradient norm about 17,649. The larger image loss is expected after correcting its reduction and cannot be compared directly to the earlier 0.215 value. A clipped KL of 3 hides the raw KL. Because your prior head is trained only through KL, its zero gradient is consistent with free nats disabling that signal; this alone does not establish posterior collapse. Weight decay can still change parameters. Decoder clipping is active, making separate versus global clipping a material choice. The file contains two step-zero blocks from different configurations, not a convergence curve.

For performance, prioritize these changes and measure their effect:

| Priority | Change | Why it matters |
|---|---|---|
| 1 | Batch image encoding across time, then batch decoding after the recurrent scan | Each image network is called 50 times per update. Flatten frames to `[B*L,C,H,W]`; only the recurrent scan must remain sequential. Separate ConvModel from the posterior head. Use microbatches if memory requires them. BatchNorm statistics change with batch composition, so decide normalization first. |
| 2 | Keep replay uint8 in RAM and preprocess sampled batches on GPU | Your uint8 conversion only affects checkpoints. At 1,005 cartpole episodes × 125 images, image payload is **5.75 GiB float32 versus 1.44 GiB uint8**. A float `[50,50,3,64,64]` batch also transfers about 117 MiB; uint8 cuts that fourfold and supports fresh dequantization per sample. |
| 3 | Detach metrics; transfer actions to CPU once per decision | [main.py:117](/home/mlarn/planet/main.py:117) accumulates graph-connected losses across 100 updates. Accumulate `v.detach()` on GPU and convert once. This retains graph metadata today, not necessarily all backward activations. Move `action.cpu().numpy()` outside the repeat loop at [main.py:163](/home/mlarn/planet/main.py:163). |
| 4 | Persist replay separately and incrementally | Every checkpoint converts and serializes all historical episodes. Separate weights/run state from replay shards; keep optional uploads outside the training critical path with bounded work and checked completion. |
| 5 | Profile, then compile stable recurrent/planner computations | Defaults perform 120 recurrent rollout steps per decision, each batched over 1,000 candidates: 15 million candidate transitions per cartpole episode. Compilation may reduce Python/kernel-launch overhead. Report compilation time separately from steady-state latency. |
| 6 | Benchmark mixed precision and transfer overlap | Training autocast is commented out; collection uses fp16. Try bf16 if supported, or fp16 with GradScaler. Keep KL/log/variance calculations, large reductions, and reward accumulation in float32; unscale before clipping. Use pinned staging/nonblocking copies when useful overlap is possible. |

The release already flattens batch and time for its [image networks](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/conv_ha.py). Compilation/transfer suggestions follow the [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html); scaling/clipping order follows [PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html). These are optimization candidates, not measured GPU speedups.

I recomputed image-network costs for your **current 16-channel defaults**:

| Image module | Your parameters | Official parameters | Your approximate MACs/image | Official approximate MACs/image |
|---|---:|---:|---:|---:|
| CNN including your projection | 307,230 | 690,144 | 18.35 million | 14.71 million |
| Image decoder | 718,467 | 3,795,555 | 18.76 million | 24.20 million |

Your full model has **1,431,438 parameters**. Combined image-network arithmetic is about **0.95×** the reference, replacing the old report's 3.60× estimate for a wider architecture. However, summed convolution/linear output sizes are about **2.50×** the reference because your networks retain larger spatial maps. Across 2,500 frames, these outputs total about 2.44 GiB in float32. This is an activation-size calculation, not measured peak allocation; backward buffers, normalization, inputs, and allocator behavior add costs. MAC counts exclude normalization, activations, biases, and backward, and approximate transpose-convolution work. I used your actual modules and a direct PyTorch translation of the released image layers.

Reduced CEM budgets, warm-started plans, multiple rollout particles, and architecture replacements are algorithmic experiments: compare evaluation return against latency and environment steps. The old CPU CEM sort and redundant wrapper rendering are already removed.

Useful plots should show control quality, reward-prediction quality over the planning horizon, and computational cost. Give W&B explicit counters for base `env_steps`, decision steps, episodes, `gradient_steps`, and elapsed time. Count seed episodes in the interaction budget and track evaluation interactions separately.

| Plot / metric | How to compute | What it reveals |
|---|---|---|
| **Evaluation return versus env steps and hours**: mean, median, percentiles | Separate full episodes without 0.3 action exploration noise; retain stochastic latents and the same planner budget. Keep evaluation out of replay. Use about 10 episodes when affordable and multiple training seeds for final comparisons. | Primary control-quality and efficiency measure. Episode variation within one model is not variation across training seeds. |
| Training return, episode length, task success | Log each fresh episode after collection and a recent-window mean. | Exploration and coverage. Current logging happens before collection and is not evaluation. `trajectories[-5:]` also handles fewer than five episodes safely. |
| **Open-loop reward RMSE/MAE/bias versus horizon**: 1/3/6/12/25/50 | Infer state from a fixed held-out context; use only the prior under recorded future actions. Compare correctly aligned targets and mask missing endpoints. | Predictive error growth, especially horizon 12. Add reward-event precision/recall or success metrics for sparse tasks. |
| **Predicted-versus-actual horizon-12 return**: scatter, bias, RMSE, ranking correlation | Compare reward sums for the same fixed continuation; optionally average stochastic rollouts. | Optimism and poor ranking that CEM can exploit. Do not compare a proposed fixed plan against a different sequence executed through replanning. |
| **Fixed video strips**: truth, reconstruction, prior prediction, error | Observe five context frames; roll out under recorded actions; show several stochastic samples on fixed clips. | Drift, blur, wrong motion, disappearing objects, and uncertainty. |
| **Raw KL, optimized KL, fraction below 3 nats**, per-dimension histograms | Log the latent-dimension sum before clipping and its distribution over batch/time. | Explains the current KL floor and zero prior gradients. Low KL alone does not prove collapse. |
| Prior/posterior std quantiles, entropy, floor-hit fraction | Aggregate over batch/time/latent dimensions, retaining tails. | Overconfidence, variance growth, inactive dimensions. Disagreement is not automatically calibrated epistemic uncertainty. |
| **Loss components and weighted contributions**, train versus held-out | Image NLL without constants, reward MSE/NLL, raw/free KL, and their exact contributions. Also plot per-pixel image MSE: `2 * obs_loss / 12288`. | Scale imbalance and overfitting. |
| **Global preclip gradient norm**, clip fraction, module norms, nonfinite counts | Measure combined norm before any clipping; retain module-level diagnostics. | Instability and domination by a head. Existing module norms are preclip values averaged across updates. |
| CEM population/elite returns and fitted std by iteration/horizon | Sample occasional calls; reevaluate retained plans with fresh rollout noise for less selection-biased estimates. | Search collapse and noisy ranking. Elite scores need not improve monotonically. |
| Proposal/executed-action bound fractions | Separate clipping of proposals from clipping after exploration. Plot action distributions. | Saturation or excessive noise; saturation can also be a valid optimum. |
| **Update/planning/rendering time**, env steps/s, GPU peak memory | Split replay/transfer, CNN, recurrence, backward, CEM, simulation, rendering, and checkpoint/upload. Use CUDA events or synchronization for GPU timing. | Actual bottlenecks; CPU timers alone can mismeasure asynchronous GPU work. |
| Replay bytes, episode count, reward histogram, sample age, successful-episode fraction | Track stored and sampled data; use fixed and optionally rolling held-out sets. | Memory growth, sparse-reward imbalance, coverage, and distribution shift. |

Videos and latent statistics have precedents in PDF Appendices H/I and the release's [prediction summaries](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/training/define_summaries.py) and [state summaries](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/summary.py). Optional frozen-latent probes to simulator angle/velocity can diagnose representation quality while keeping privileged state out of agent training and planning.

Start with the bolded plots, evaluate every 10–25 collected episodes initially, and save videos less frequently. Restore model modes after validation and isolate evaluation environments/RNG streams. Label horizons in decision steps and base steps: cartpole's horizon 12 at repeat 8 covers 96 base steps.

Validation performed: CPU forward/backward through all five actual modules with batch size 2 and sequence length 3; finite losses/gradients; actual-model CEM execution; numerical KL comparison; synthetic reward alignment; restored BatchNorm behavior; bounded proposals, configurable K, and the K=1 failure; new replay round-trip and old-format misdecoding; short replay behavior; actual CLI parsing; real reward-type checks for all six environments; cartpole reset-after-terminal behavior; and current network counts. Explicit CUDA allocations were redirected to CPU only inside the temporary check harness. Training source files were unchanged.

CUDA was unavailable to PyTorch in this session. I did not measure GPU throughput/peak memory, run full training, render validation videos, or execute the legacy TensorFlow implementation. The supplied logs cannot establish convergence or reproduction of paper scores. Fix task/configuration/replay issues, establish separate evaluation with the intended observations, align objective/CEM/optimizer defaults, then profile and ablate architectural changes one at a time.
