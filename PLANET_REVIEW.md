Review date: 2026-09-28. Local revision: `b7ec8529d6e58b33e220849e0982d42363abbe84`.

Your implementation captures the main PlaNet algorithm, but several details prevent it from being a faithful reproduction and can materially impair learning. Fix the likelihood scaling, training/evaluation modes, action bounds, and observation setup before interpreting long training runs or tuning the model. The largest avoidable computational costs are image processing, redundant rendering, and the CPU work inside CEM.

I reviewed the supplied [PLANET.pdf](/home/mlarn/planet/PLANET.pdf), including its appendices, every executable source file in this repository, the saved debug dataset's configuration, the available training log, and the authors' [official implementation](https://github.com/google-research/planet/tree/c04226b6db136f5269625378cd6a0aa875a92842). The supplied PDF is arXiv v5, dated June 4, 2019. The official code is pinned to `c04226b6db136f5269625378cd6a0aa875a92842`; its README explicitly notes changes since the paper. Below, “official” means that released code, and “paper” means your PDF. They are not identical.

**Correctness findings, in priority order**

1. **High: image reconstruction has the wrong scale relative to KL and reward.** [utils.py:81](/home/mlarn/planet/utils.py:81), [utils.py:110](/home/mlarn/planet/utils.py:110), [utils.py:118](/home/mlarn/planet/utils.py:118).

   `nn.MSELoss()` averages over batch, channels, height, and width. The official image likelihood sums over the 12,288 image coordinates, then averages over batch and time. For a unit-variance Gaussian, its parameter-dependent loss is `0.5 * sum(pixel_squared_error)`. Your image term is therefore **6,144 times smaller relative to KL** than the official likelihood term. Your scalar reward MSE also omits the Gaussian factor of one-half, making the reward-to-image weighting differ by 12,288 times from the released objective. This is a substantive objective change, not a harmless reporting convention. Sources: [official decoder](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/conv_ha.py), [objective reductions](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/training/utility.py), [PyTorch MSE reduction](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MSELoss.html).

   For one time step, match the released objective, up to constants, with:

   ```python
   obs_nll = 0.5 * (pred_obs - observation).square().flatten(1).sum(-1).mean()
   reward_nll = 0.5 * (pred_reward.squeeze(-1) - reward).square().mean()
   loss = obs_nll + 10.0 * reward_nll + kl.clamp_min(3.0).mean()
   ```

   Average over time as well; mask missing reward targets and document their denominator. Keep per-pixel MSE as a separate readable diagnostic. `reward_scale=10` itself matches the released code. Your existing log contains `obs_loss≈0.215`, weighted reward loss `≈34.18`, and KL `≈3.81`, consistent with this imbalance. That single logged update block does not establish a learning trend. Correcting the loss will change its numerical scale sharply; compare control returns and validation diagnostics across the change.

2. **High: models stay in evaluation mode after the first collected episode.** [main.py:89](/home/mlarn/planet/main.py:89), [main.py:140](/home/mlarn/planet/main.py:140).

   `.train()` runs once before the outer loop. Collection calls `.eval()`, but the next optimization phase never restores `.train()`. Both the encoder and image decoder contain BatchNorm. After the first collection phase, subsequent optimizer updates use frozen running statistics while their weights continue changing. Evaluation mode does not disable gradients, which makes this easy to miss. A component probe confirmed that BatchNorm's tracked batch count stopped increasing on the next update. Restore training mode at the start of every fitting phase. For closer reproduction, remove BatchNorm: the official convolutional networks have none. [BatchNorm behavior](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm2d.html).

3. **High: actions are unbounded during planning and after exploration noise.** [cem.py:33](/home/mlarn/planet/cem.py:33), [main.py:157](/home/mlarn/planet/main.py:157).

   Neither sampled CEM sequences nor the executed noisy action are clipped. CEM can select trajectories outside the legal action range, where the learned model need not be reliable. Internal actuator limiting is not a substitute for giving planning, replay, and the recurrent update the same bounded action. A simple increasing-reward probe returned an action of `2.32` for a nominal `[-1, 1]` task; the first proposal population reached absolute values above `4`.

   Bound proposals before evaluating and refitting them, then bound the executed action after adding exploration noise. Store and propagate that exact executed action. Infer action dimension and limits from `env.action_spec()` before constructing the models. Currently `action_dim=1` is an independent CLI default, so changing the task can also produce incompatible model inputs. The official [planner](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/planning.py) and [MPC agent](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/mpc_agent.py) both enforce bounds.

4. **High for benchmark fidelity: observations differ from the official environment setup.** [main.py:36](/home/mlarn/planet/main.py:36), [utils.py:128](/home/mlarn/planet/utils.py:128).

   `visualize_reward=True` changes object colors to indicate reward. Those colors enter the agent's observations, changing the visual learning problem and potentially providing a reward-related shortcut. The official environment leaves this disabled. Your pixel wrapper also uses the installed renderer's defaults: `240×320`, free camera `-1`, followed by a resize to `64×64`. The official code uses `64×64` rendering and camera `0`. Configure these explicitly, and compare actual task images before benchmarking. The official setup is in [tasks.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/tasks.py); local installed dm_control source confirms the defaults and reward visualization behavior.

5. **Medium: training and collection initialize the recurrent state differently.** [utils.py:90](/home/mlarn/planet/utils.py:90), [main.py:146](/home/mlarn/planet/main.py:146).

   Training first computes `h = GRU(s=0, a=0, h=0)` and then infers the posterior. Collection infers its first posterior directly from `h=0`. With GRU biases these differ; an initialization probe measured a maximum absolute difference of `0.0513`. Use a shared observation/filtering step for training and acting. The official MPC agent performs the zero-action transition before processing the first observation. This is an initial-state mismatch; your recurrent/action ordering on subsequent collection steps is otherwise consistent.

6. **Medium: CEM ignores `K` and fails for smaller candidate populations.** [cem.py:39](/home/mlarn/planet/cem.py:39).

   Elite selection, expansion, and denominator hardcode `100`, `100`, and `99`. A probe changing `K` from 100 to 5 produced identical actions under the same seed; 32 candidates caused a shape error. Use `args.K` everywhere, validate `1 <= K <= candidates`, and select elites with `torch.topk`. The spread-estimator difference is discussed separately below because your formula actually matches the paper's pseudocode.

7. **Medium: episode termination and sequence length are assumed rather than checked.** [main.py:74](/home/mlarn/planet/main.py:74), [main.py:152](/home/mlarn/planet/main.py:152), [dataset.py:20](/home/mlarn/planet/dataset.py:20).

   The repeated-action loops never inspect `time_step.last()`. In dm_control the next step after termination resets the environment, so an episode can silently cross a reset if a task terminates early, `T` exceeds its limit, or the last action repeat overruns it. The default 1,000-step cartpole task with repeat 8 happens to divide evenly. Stop both collection loops at termination and sample chunks from each stored episode's actual length. Reject short episodes or pad and mask them explicitly; keep terminal observations.

8. **Medium: the logged reward is wrong and is not an evaluation score.** [main.py:130](/home/mlarn/planet/main.py:130).

   `[-i for i in range(5)]` addresses episodes `[0, -1, -2, -3, -4]`: the first seed episode stays in the average, replacing the fifth most recent episode. With fewer than five trajectories, it can fail. Use `dataset.trajectories[-5:]` for a recent training-return statistic. Logging occurs before collecting the current episode, so that score also lags the current model. All planned collection includes exploration noise; add separate evaluation episodes without exploration noise or replay insertion. Log the freshly collected training return directly and label training and evaluation separately.

9. **Medium: debug defaults and argument parsing make runs easy to misconfigure.** [main.py:62](/home/mlarn/planet/main.py:62), [main.py:220](/home/mlarn/planet/main.py:220), [main.py:227](/home/mlarn/planet/main.py:227).

   `debug=True` loads a pickled dataset object instead of collecting seed episodes. Its saved batch size, sequence length, and action settings replace the freshly configured dataset. The supplied pickle contains 5 episodes, length 125, batch size 50, sequence length 50, action dimension 1, and repeat 8; it has no task metadata to validate a requested domain. For example, changing CLI batch size while leaving debug enabled makes dataset and loss dimensions disagree. Default to normal collection and make replay import an explicit, validated operation.

   `type=bool` also parses `--resume False` and `--setup_wandb False` as true. Use `BooleanOptionalAction` or `store_true`/`store_false`. `setup_dirs()` recursively deletes existing directories for a reused run ID when not resuming ([utils.py:32](/home/mlarn/planet/utils.py:32)); use a new run directory or fail clearly on collisions. The review did not execute this setup or the training entry point.

10. **Medium for reproducibility and recovery: environment seeds and checkpoint metadata are incomplete.** [main.py:36](/home/mlarn/planet/main.py:36), [main.py:175](/home/mlarn/planet/main.py:175), [utils.py:18](/home/mlarn/planet/utils.py:18).

    Seeding NumPy's global RNG does not seed the task's independently created `RandomState`; pass `task_kwargs={"random": args.seed}`. Save configuration, environment-step counters, Python/NumPy/Torch RNG states, and the task RNG state for reproducible episode-boundary resumes. Current checkpoints preserve model and optimizer weights but cannot exactly resume the sampling trajectory. Add a final checkpoint: with 1,000 outer iterations and save interval 50, the last periodic checkpoint is step 950, not 999. HF upload is unconditional at save time and its future is immediately awaited; make remote publishing optional and separate it from successful local persistence.

**What is already correct, including indexing subtleties**

- Your deterministic state, stochastic Gaussian prior, observation-conditioned posterior, reparameterized sample, and observation/reward decoders implement the RSSM factorization. All observation information reaches later dynamics through the sampled state; there is no deterministic encoder-to-future shortcut.
- The analytic KL is `KL(posterior || prior)` with the right standard-deviation terms and sum over latent dimensions. A numerical comparison to PyTorch distributions agreed within `6.2e-5` absolute error in float32. Summing latent dimensions before applying 3 free nats is appropriate. The released `max(KL - 3, 0)` differs from your `max(KL, 3)` only by a constant, with the same gradients away from the threshold. A logged clipped KL of exactly 3 does not imply a raw KL of 3.
- Your replay stores `(observation_before_action, action, reward_after_action)`. Consequently, at chunk index `k > 0`, predicting `reward[k-1]` from the state inferred from `observation[k]` is aligned correctly. Changing it to `reward[k]` without changing storage would introduce a bug. The official replay instead stores an initial dummy action/reward and then `(resulting_observation, action_taken, reward_received)`, so its targets use matching indices. [Official episode convention](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/wrappers.py).
- There is nevertheless an endpoint loss: each length-50 chunk trains only 49 reward transitions and divides their sum by 50, and the final transition of each episode is never trained because its resulting observation is not saved. Store `T+1` observations for `T` transitions, or adopt the official dummy-initial-row convention. This matters especially for rewards near episode end.
- Planning rolls forward with the prior, sums future reward means, executes the first fitted action mean, and replans after a real observation. It does not decode images while planning. Using one stochastic rollout per candidate is intentional in the paper; multiple particles are an optional experiment, not a missing requirement.
- **Missing overshooting is not a bug.** Appendix A says the final agent no longer required overshooting or a fixed global prior, and Appendix D reports slightly worse RSSM performance with overshooting. Both are disabled in the released defaults. Add overshooting only as a measured ablation.
- Restarting CEM from zero mean/unit variance at each environment decision is faithful. Warm-starting the previous plan would change the method and can change local-optimum behavior. There is no missing actor, critic, discount factor, or terminal value in this finite-horizon baseline. An unbounded image decoder mean is also appropriate for its Gaussian likelihood.

**Differences between your implementation, the paper, and the released code**

| Component | Your implementation | Paper / official release | Assessment |
|---|---|---|---|
| RSSM sizes | Deterministic 200, stochastic 30 | Same | Matches |
| Image encoder | Five 3×3 padded convolutions; first stride 1; BatchNorm; linear compression to 30 | Official: four 4×4 stride-2 valid convolutions; 1,024 image features; no BatchNorm | Significant capacity and compute difference |
| Posterior head | Concatenate 30 image features with 200 hidden features; hidden width 230 | Official: concatenate 1,024 image features with 200 hidden features; hidden width 200 | Your 30-dimensional image bottleneck is an extra restriction before the stochastic bottleneck |
| GRU input | Direct concatenation of state and action | Official: learned dense/ReLU projection to 200 before the GRU | Valid variant, not identical transition architecture |
| Prior head | Two 200-unit hidden layers | Official default: one 200-unit hidden layer | Capacity difference |
| Reward head | Two 230-unit hidden layers | Paper describes two 200-unit layers for feed-forward functions; released reward head uses three 300-unit layers | The paper and release differ too |
| Image decoder | Dense to 256×4×4; four 3×3 transpose convolutions with BatchNorm; final convolution | Official: dense to 1,024×1×1; transpose-convolution kernels 5, 5, 6, 6; no BatchNorm | Yours has fewer decoder weights but more arithmetic and larger activations |
| Latent standard deviation floor | 0.01 | Official default: 0.1 | More concentrated beliefs and potentially larger KL gradients; ablate only after correctness fixes |
| Image likelihood | Per-pixel mean squared error | Sum over image dimensions in Gaussian log likelihood | Major loss-scaling mismatch described above |
| Reward weighting | 10× MSE | Released code: 10× Gaussian NLL | Weight 10 matches; Gaussian factor differs |
| Optimizer | Adam, lr 1e-3, default epsilon 1e-8, weight decay 1e-4 | Paper/release: lr 1e-3, epsilon 1e-4; official optimizer has no weight decay | Restore these for a reproduction baseline |
| Gradient clipping | Each of five modules separately at norm 1,000 | Official: one global norm across parameters at 1,000 | Different when clipping activates; global norm can remain above 1,000 in your version |
| CEM settings | Horizon 12, 10 iterations, 1,000 candidates, hardcoded 100 elites | Same default settings | Matches at defaults; configurable K is broken |
| CEM spread | Sum of absolute deviations divided by K−1 | Paper Algorithm 2 prints this; official code uses `sqrt(population_variance + 1e-6)` | Source discrepancy, not evidence that you misread the paper |
| Action bounds | Unbounded proposals and noisy executed actions | Official clips both | Correctness issue |
| Replay preprocessing | Float32 image with dequantization noise fixed at collection | Official saves uint8 images and adds fresh dequantization noise when sampling training batches | More memory and different noise augmentation |
| Replay sampling | Uniform episodes and random windows | Paper specifies uniform chunk sampling; released recent loader balances recent and cached episode pools | Sampling-policy difference; your equal-length replay is broadly consistent with the paper |
| Training cadence | 100 updates followed by one episode | Paper: same; official `collect_every=5000` counts increments of batch size 50 | Do not mistake the official value for 5,000 optimizer updates |
| Seed episodes / chunk shape | 5 episodes; batch 50; length 50 | Same training defaults | Debug dataset path can override the intended setup |
| Evaluation | No separate evaluation loop | Official has separate test episodes, zero exploration noise, and prediction diagnostics | Needed for meaningful performance comparison |

Architecture and configuration sources: [official RSSM](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/models/rssm.py), [image networks](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/conv_ha.py), [head networks](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/networks/basic.py), [configuration](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py), [optimizer](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/custom_optimizer.py), [replay](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/numpy_episodes.py).

For an official-code baseline, use the variance-based CEM update. It is the Gaussian maximum-likelihood fit and includes a floor that prevents the search variance from becoming exactly zero. In PyTorch, the corresponding update is:

```python
elite_idx = returns.squeeze(-1).topk(K, sorted=False).indices
elite = bounded_actions.index_select(0, elite_idx)
mean = elite.mean(0)
std = (elite.var(0, correction=0) + 1e-6).sqrt()
```

Action repeats must also be task-specific: cartpole 8; reacher, cheetah, and cup 4; finger and walker 2. The same horizon of 12 therefore represents different durations. Your default is appropriate for cartpole, but should not silently carry over to every task. [Official tasks](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/tasks.py).

**Performance opportunities, ordered by expected usefulness**

The following are source-based opportunities and arithmetic estimates, not measured GPU speedups.

| Priority | Change | Why it matters here |
|---|---|---|
| 1 | Render once per decision, directly at 64×64, camera 0 | `pixels.Wrapper` renders on every `env.step`, including all eight repeated steps. A mocked-render probe confirmed eight render calls per cartpole decision. Put action repetition inside the pixel wrapper, or step the base environment and render only the final observation. Break at termination. This removes 7/8 render calls at repeat 8, while the smaller resolution has 18.75× fewer pixels than 240×320. These are operation counts, not guaranteed timing ratios. |
| 2 | Replace CPU sorting and tensor unbinding with GPU `topk`/gather | `rewards.cpu().numpy()` synchronizes every CEM iteration, and Python sorts all 1,000 proposals then stacks 100 separate tensors. At default cartpole settings this runs 1,250 times per collected episode. Keep scoring, selection, and refitting on the GPU. |
| 3 | Use the official image network as a baseline | The high-resolution feature maps in your architecture cost more than its parameter count suggests. See the arithmetic comparison below. Removing BatchNorm also removes the current mode-dependent architecture difference. |
| 4 | Batch image encoding and decoding across time, with memory-aware microbatches | `compute_loss` launches each image network 50 times per update. Image features do not depend on recurrent state: encode `[B*T,3,64,64]`, run the recurrent posterior/prior scan, then decode stacked features and compute reward heads in batches. The official code already flattens batch and time for its image networks. With BatchNorm, flattening time changes statistics, so first choose the normalization behavior. On small GPUs, use microbatches/checkpointing or accumulated smaller sequence batches rather than blindly processing all 2,500 frames at once. |
| 5 | Keep replay as uint8 and preprocess on the GPU when sampled | The default run stores 1,005 episodes × 125 frames × 3×64×64 float32 values: **5.75 GiB** for images alone. Uint8 reduces this to **1.44 GiB** and permits fresh dequantization noise per sample. At repeat 2, the float32 image payload grows to about 23 GiB. Store task/configuration metadata with replay. |
| 6 | Avoid saving and uploading the full replay inside every model checkpoint | Current saves repeatedly serialize all growing trajectories, and `.result()` blocks on upload. Persist replay separately/incrementally; checkpoint weights, optimizer, RNG, configuration, and replay references. Keep remote upload outside the critical update/collection path and handle its result explicitly. |
| 7 | Detach metric accumulation and move the action to CPU once per decision | `metrics_dict[k] += v` retains autograd references across 100 updates; use detached scalars/tensors. This needlessly retains graph metadata, although backward normally frees saved activation buffers, so it is not proof of a complete activation-memory leak. Likewise, `action.cpu().numpy()` is inside the repeat loop; transfer once, reuse it, and retain the same bounded action for replay. |
| 8 | Profile before compilation, mixed precision, or reducing planning budget | Time replay sampling/transfers, encoder, recurrent scan, decoder, backward, CEM, environment simulation/rendering, and checkpoint/upload separately. After the larger fixes, try `torch.compile` on the recurrent/planning computation and appropriate mixed precision. Keep KL/log/variance calculations and long reward accumulation in float32. Training AMP is currently commented out; collection already uses float16 autocast. FP16 training requires appropriate scaling and unscaling before gradient clipping. Hardware determines whether AMP is faster. |

Pinned memory/nonblocking copies and compilation are standard options described by the [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html); benchmark them here after eliminating the explicit CPU synchronizations. Do not promise a speedup from a flag alone. Reducing CEM candidates/iterations/horizon changes the search budget and must be evaluated as a return-versus-latency tradeoff. Warm starts and multiple rollout particles are also algorithmic ablations.

Arithmetic comparison at batch size 1:

| Image module | Your parameter count | Official parameter count | Your approximate MACs/image | Official approximate MACs/image |
|---|---:|---:|---:|---:|
| Convolutional encoder including your projection | 1,102,878 | 690,144 | 69.72 million | 14.71 million |
| Image decoder | 1,925,379 | 3,795,555 | 70.54 million | 24.20 million |

These counts use your actual modules and a direct PyTorch translation of the released image network. They count convolution/linear multiply-accumulates, exclude activations, normalization, biases, and backward, and approximate transpose-convolution work. The combined image networks perform about **3.60×** the nominal arithmetic despite the smaller decoder. Your complete model has **3,433,998 parameters**. The summed convolution/linear output sizes alone across a batch of 50 sequences of length 50 are about **4.77 GiB** in float32, before accounting for other backward requirements. This is an activation-size estimate, not measured peak allocation; memory pressure is a credible concern on a 6 GiB GPU.

**Useful metrics and plots during training**

Use separate x-axes/counters for `env_steps` (calls to the base environment before action repetition), `episodes`, `gradient_steps`, and elapsed hours. Your current `step` is one block of 100 updates and one collection, so it obscures both sample efficiency and compute efficiency. Count seed episodes in the interaction budget and record evaluation interactions separately.

| Plot / suggested metric | How to compute | What it diagnoses |
|---|---|---|
| `eval/return_mean`, median, distribution across episodes | Periodically run approximately 10 full episodes without exploration noise, using the same planner settings; keep them out of replay. Plot against environment steps and wall-clock time. | Actual control quality and learning efficiency. Use multiple training seeds for final comparisons; variation across episodes of one model is not a substitute for variation across trained seeds. |
| `train/episode_return`, recent mean, episode length, task success | Log immediately after collection, separately from evaluation. | Exploration cost, early termination, and sparse-reward discovery. Raw sums of repeated rewards preserve the task's usual return units. |
| `val/posterior_image_mse` and `val/prior_image_mse` | Decode posterior reconstructions and one-step prior predictions on held-out episodes. | Separates reconstruction ability from predictive dynamics. Good reconstructions alone do not establish a useful planner. |
| `val/openloop_reward_rmse_h{1,3,6,12,25,50}` | Infer state from a fixed observation context, then roll only the prior under recorded held-out actions. Compare predicted rewards to aligned ground truth. Aggregate multiple stochastic trajectories where practical. | Error growth over the planning horizon, especially horizon 12. Also report MAE and bias; constant predictions can have deceptively small MSE on sparse rewards. |
| `val/openloop_return_bias_h12`, RMSE, predicted-versus-actual scatter | Sum predicted rewards along a fixed 12-action held-out continuation, compare with the sum actually obtained under those same actions. | Whether the model systematically overestimates candidate plans. CEM can exploit optimistic errors. Do not compare a proposed fixed plan with rewards from a different, repeatedly replanned action sequence. |
| Ground truth / reconstruction / open-loop video strips | Observe the first 5 frames; reconstruct them and then predict future frames under recorded actions, showing several stochastic samples and an error image. | Blur, drift, object disappearance, wrong dynamics, and posterior collapse. Use fixed clips across checkpoints. This follows the paper's Appendix H and the official open-loop context of 5. |
| `latent/kl_raw`, `latent/kl_free`, `latent/fraction_below_free_nats` | Log mean raw `sum_d KL(q_d||p_d)`, its clipped loss, and the fraction below 3 nats. Add per-dimension KL histograms. | Reveals behavior hidden by the clipped KL plateau. Low raw KL can mean good agreement or underused stochastic state; interpret it with predictions and control. |
| `latent/prior_std`, `posterior_std`, entropy, std quantiles | Summarize distributions over batches/time/dimensions; track mass near the chosen floor. | Overconfidence, variance explosions, and inactive dimensions. Posterior/prior difference is not by itself a calibrated estimate of epistemic uncertainty. |
| `loss/image_nll`, `reward_nll`, raw/free KL, weighted contributions | Log each unweighted component and the exact contribution to the optimized total. Retain image MSE per pixel for readability. | Scale imbalance such as the current image-loss issue. Add train-versus-held-out curves to detect memorization. |
| `optim/global_grad_norm`, clip fraction, per-module norms, nonfinite counts | Measure global norm before clipping and record whether clipping activated; keep module norms for diagnosis. | Unstable updates and domination by one head. Your existing module norms are preclip norms, which are useful, but there is no global norm statistic. |
| `planner/elite_return`, population mean, action std by horizon/iteration | Log sampled CEM diagnostics on occasional decisions, not every candidate of every call. Reevaluate a retained plan with fresh samples if assessing improvement. | Search collapse and noisy/optimistic ranking. With stochastic rollouts, elite values need not increase monotonically across iterations. |
| `planner/action_bound_fraction`, preclip action magnitude | Count coordinates clipped in proposals and executed noisy actions separately. | Bound violations and saturation; high saturation can also be a valid optimum, so interpret alongside return. |
| `perf/update_ms`, `plan_ms_p50/p95`, `render_ms`, `env_steps_per_sec`, memory and replay bytes | Use CPU timers for environment/I/O and CUDA events or appropriate synchronization for GPU timing. Split checkpoint time from training. | Identifies the actual bottleneck and makes speed claims reviewable. |
| Replay size, reward histogram, nonzero-reward fraction, replay age | Track episodes, decisions, successful episodes, and sampled data age. | Insufficient coverage, sparse-reward imbalance, and stale data. Separate a fixed validation set from a rolling held-out set when the policy distribution changes. |

For sparse tasks, add reward-event precision/recall or another task-specific success metric so an always-zero reward model cannot look good merely from class imbalance. For cartpole, optional simulator-state diagnostics can measure angle/velocity prediction from frozen latents; keep those probes separate from model training and planning if preserving the pixels-only setting. The paper uses this kind of diagnostic in Appendix I. The released implementation already logs prior/posterior entropy and standard deviation, prediction errors, open-loop videos, and test returns: [state summaries](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/summary.py), [training summaries](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/training/define_summaries.py).

Start with a small dashboard: evaluation return, training return, held-out reward error versus rollout horizon, a fixed open-loop video, raw KL plus fraction below free nats, weighted loss components, global gradient norm, action saturation, and update/planning/rendering time. Compute scalars frequently, evaluate every 10–25 collected episodes initially, and save videos less frequently. Restore model modes after validation and isolate evaluation random-number streams so diagnostics do not unnecessarily perturb training.

**Validation performed and limits**

I ran an isolated CPU forward/backward check with the actual five model modules using batch size 2 and sequence length 3; losses and gradients were finite. The check redirected explicit CUDA tensor allocations to CPU without changing the loss equations or repository source. I numerically checked the KL formula and image-loss ratio, reproduced the BatchNorm mode problem, nonzero zero-input GRU initialization, ignored K, small-candidate CEM failure, unbounded CEM output, boolean parsing, and reward-window indexing. I inspected the supplied debug replay shapes/dtypes and derived the arithmetic/parameter counts above. A real dm_control cartpole instance with rendering mocked out confirmed the wrapper's render-call count, default image shape, and reset-after-terminal behavior.

This is a code and component review, not a full training reproduction. PyTorch reported CUDA unavailable in this tool session, so I did not measure GPU throughput or peak memory, run the full training loop, or execute the legacy TensorFlow implementation. There is only one relevant update block in the current plain-text training log, which is insufficient to evaluate convergence. No training source files were changed. Recommended order: fix correctness and evaluation first, establish a reproducible cartpole baseline, remove measured overhead, then ablate architectural or planner changes one at a time.
