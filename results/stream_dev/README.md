# Gridworld task stream: dev runs (NOT the official pilot)

The official pilot on seeds 500-509 is ON HOLD until the metric and seeds are recorded in
tars-test-rules.md. Nothing has run on 500-509. Every run below is a dev/debug run on 900+ seeds.

| file | seeds | what | config |
|---|---|---|---|
| smoke.npz | 900-901 | pipeline smoke test | lifelong_stream, 4 new tasks, K=20, revisit_every=3, stream_seed=0, fresh baselines on |
| size_k60.npz (+ size_k60/ plots, summary.json) | 910-919 | throughput sizing + first look | lifelong_stream, 6 new tasks (8 blocks), K=60, revisit_every=3, stream_seed=0, fresh on |
| (tests) | 900-919 | pytest uses dev seeds 900-919 as stream/agent seeds (fresh 1900-1919) | tiny configs |
| timing_1m_s930.json/.log | 930 (fresh 1930) | timing only, 1 seed | lifelong_stream, hidden 905 (999,125 params), 3 new tasks + 1 revisit (4 blocks), K=100, fresh on all 3 tasks |
| timing_1m_10seeds.json/.log | 930-939 (fresh 1930-1939) | timing + memory, 10 seeds vmapped | lifelong_stream, hidden 905, 2 new tasks (2 blocks), K=25, fresh on both tasks |
| (layouts only, no training) | 500-509, 900-939 | demo-pool overlap test generates these streams at pilot length (40 new tasks); no agent is run | stream.reference_streams() |
| (layouts only, no training) | 500-509 | demo-pool overlap test also generates the 200-task null-run streams for these seeds (2026-10-25); no agent is run | stream.NULL200_STREAM |
| (no file; eval timing) | 930-939 | 2026-10-25: timed `ppo.evaluate` (greedy, all stream tasks, max_steps 64) with random-init hidden-950 params on the 200-task dev layouts, 10 seeds vmapped, nice -n 10, no training: 0.71 s/block at T=40, 3.24 s/block at T=200 | /tmp/eval_cost.py (for the null-200 time estimate) |
| null200_geom_k2_940_949.log | 940-949 (fresh 1940-1949) | 2026-10-25 17:31-18:43 GMT, nice -n 10: timing + RSS at full 200-task geometry, K=2 updates/block, head_probe_every 5 (40 snapshots). Stream finished: 266 blocks in 2,606.7 s (incl. compile). Killed during fresh (not needed: per-task fresh cost is independent of T). Peak RSS 3.52 GB (ps sampling every 5 s). No npz. | lifelong_stream, hidden 950, 200 new tasks, revisit 3, matmul highest |
| null200_geom_k2_T40_940_949.json | 940-949 (fresh 1940) | 2026-10-25 18:44-18:51 GMT, nice -n 10: same at 40 tasks via time_stream.py: stream 53 blocks run 389.8 s (compile 5.3 s), peak RSS 1.83 GB; 1 fresh task | lifelong_stream, hidden 950, K=2, revisit 3 |

Fresh-seed note: smoke.npz and size_k60.npz were made at f798aa8, before the paired fresh-seed
block existed. Their fresh agents used key fold_in(PRNGKey(stream seed), 1_000_000 + task index).
From the follow-up commit on, fresh agents use fresh_seed = stream seed + 1000 for dev seeds
(+100 for official 500-509 -> 600-609), with task numbers folded in from 1.

Bit-identity reference for the ppo.py refactor: seeds 900-901 (scratch file outside the repo).

## Sizing (10 seeds, vmapped, 8-core CPU, while other jobs were running)
* stream: 8 blocks x 60 updates (2048 env steps/update) = 480 updates in 114 s incl. compile, ~0.22 s/update for all 10 seeds
* fresh baselines: 6 tasks x 60 updates in 106 s incl. compile, ~0.29 s/update

## First look (size_k60, seeds 910-919, thr = 0.8, M = K = 60 updates; dev only, do not cite as the result)
Tasks are numbered from 1. Metric names follow tars-test-rules v2 (plots regenerated with the new code).
* Fresh agents learn a task in ~40-60 updates (mean normalised return 0.94 at update 60; 5% capped at thr 0.8).
* Stream agent: large drop vs fresh from task 2 on. Late-stream AUC_rel IQM -0.49, CI [-0.59, -0.37].
  Policy entropy at the switch collapses from 1.39 (task 1) to ~0.2-0.35. Dormant fraction 0.20 -> 0.23,
  srank(trunk_1) 73 -> 68.5, weight norm up; the random-target probe shows no trend.
* PRIMARY (AUC_rel slope, new tasks, task 1 excluded): +0.020, CI [-0.085, 0.103]. No progressive slope in this
  6-task stream.
* Secondary: slope including task 1 = -0.065 [-0.118, -0.013]; primacy bias (AUC_rel task 2 - task 1) = -0.45
  [-0.83, -0.09]. The drop is a one-off at the first switch.


## Timing at ~1M params (development CPU, 8 vCPU Xeon, 15 GB RAM, jax/jaxlib 0.12.0.1, cpu; 2026-10-24, nothing else running)
lifelong_stream preset, shared trunk, hidden 905 -> 999,125 params (P = h^2 + 199h + 5; hidden 950 -> 1,091,555).
2048 env steps/update (32 envs x 64 steps), 4 epochs x 4 minibatches. Per-block eval and probes are included in the run time.

| | seeds | JIT compile | s/update (all seeds) | s per seed-update | peak RSS |
|---|---|---|---|---|---|
| stream | 1 (930) | 5.7 s | 0.328 | 0.328 | 0.80 GB |
| fresh | 1 (1930) | 5.6 s | 0.356 | 0.356 | 0.93 GB (process) |
| stream | 10 (930-939) | 5.4 s | 2.63 | 0.263 | 1.60 GB |
| fresh | 10 (1930-1939) | 5.4 s | 2.95 | 0.295 | 1.80 GB (process) |

10 seeds vmapped fit easily (1.8 GB peak of 15 GB), so this was measured, not extrapolated. Cost scales
about linearly with seeds on CPU (10 seeds = 8x one seed).

Extrapolation, 40-task pilot (53 blocks + 40 fresh tasks, K=100, 10 seeds vmapped). Compile time is negligible (3 x ~6 s).
* hidden 905 (1.0M): stream 5300 updates x 2.63 s = 3.9 h; fresh 4000 x 2.95 s = 3.3 h; head-reset probe
  8 tasks x 100 updates <= 0.65 h (upper bound at the fresh rate, not measured; the trunk is frozen). Total ~7.2 h, or ~7.8 h with the probe.
* hidden 950 (1.09M, as pinned in scripts/run_pilot.sh): x~1.09 (FLOPs scale with params) -> ~7.8 h, or ~8.5 h with the probe.
  The probe adds ~0.35 GB for the 8 param snapshots of 10 seeds.
* Per stage-1 cell (one arm, 10 seeds, same 40-task stream, 1M): stream only ~3.9 h (4.2 h at 950).
  The fresh baselines (+3.3 h) can be shared by all cells with the same architecture, LR and stream, because they now
  always use the scratch config (B-B3). Otherwise ~7.2 h per cell. BC pretraining for the warm-start arms is not included (not measured).

RTX 3060 12 GB (ESTIMATE, not measured: no GPU on the development machine). About 53 GFLOP per seed-update at 1M
(rollout forward 2048 x 2P plus 4 epochs x 2048 x 6P), so 0.53 TFLOP per update for 10 seeds. The CPU achieves
~0.2 TFLOP/s. A 3060 (12.7 TFLOP/s fp32 peak) at 30-50% utilisation on these small matmuls, plus the sequential 64-step
rollout and launch overheads, gives roughly 0.15-0.3 s/update for 10 seeds, about 10-15x the CPU. That puts the
pilot at ~0.5-1 h and a stream-only cell at ~15-30 min. Memory is not a constraint (~2 GB on CPU). Caveat: on
Ampere, JAX uses TF32 for fp32 matmuls by default. Use jax_default_matmul_precision=highest for the 1e-5
equivalence check (numerical_equivalence.py --matmul-precision highest --tol 1e-5), which costs some of the speed-up. Measure before planning on it.
