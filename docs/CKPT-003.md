# CKPT-003 — scale, long context, and theory (planned)

**Target label:** `v1.0.0-scale`  
**State:** planning and local protocol infrastructure. **No 100M/300M/1B model,
1M-token execution, scaling curve, RAG comparison, or theorem is claimed by
this repository at this state.** CKPT-002's benchmark exit criteria remain
independent and incomplete.

## Evidence-first progression

A CKPT-003 run is blocked until it has an immutable data revision and hash,
tokenizer hash, code SHA, complete topology receipt, trainable parameter count,
FLOP accounting method, precision, effective distributed configuration, seed,
loss trace, and checkpoint SHA-256. A configuration file, a dry-run, and an
allocated cluster are not a completed run.

The intended escalation is deliberately gated:

1. single-device shape/checkpoint/resume smoke;
2. two-worker numerical-parity and failure-recovery test;
3. a small distributed run with telemetry receipts;
4. a 100M series; then 300M; then 1B only after the preceding stage passes;
5. 4K, 16K, 64K, 1M context stages with the same failure gates.

This repository cannot execute the requested 4×A100 80GB or 8×V100 runs in
this environment. Cloud credits, approved data, storage, and an independent
reproduction team are external dependencies, not completed artifacts.

## Distributed training recipe

Two non-executable, version-neutral templates are checked in:

- `configs/ckpt003/deepspeed_zero3_bf16.json` — ZeRO-3/bf16/checkpointing.
- `configs/ckpt003/fsdp_recipe.yaml` — torchrun/FSDP/fallback and mandatory
  run-receipt fields.

DDP, FSDP, and DeepSpeed must be compared only after fixing architecture,
tokenizer, data order, global batch, optimizer schedule, target token count and
precision. Capture actual PyTorch, CUDA, NCCL, DeepSpeed, driver, GPU, world
size, host topology and interconnect versions. Prefer bf16 where hardware
supports it; fp16 requires dynamic-loss-scale/overflow telemetry. FlashAttention
is optional and must record kernel/version and numerical-parity checks against
the stated fallback before performance claims.

## Scaling-law contract

`hga.evaluation.scale_protocol.fit_power_law` accepts only completed,
SHA-identified observations and fits:

\[
 L(x) = a x^b,\quad x\in\{P,\mathrm{training\ FLOPs}\}.
\]

It requires a fixed training-token budget, dataset revision and code revision
within each fit. The reported R² is **OLS R² in log-loss space**. The preregistered
screen is R² ≥ 0.95, but a pass does not prove causality or safe extrapolation;
a fail is reported as a result, not removed. At least three completed points are
required; the requested 100M, 300M, and 1B series are the minimum target points.

## 1M-token stability contract

`assess_long_context_stability` evaluates one actual run at a time and refuses
to call an empty/undersampled trace stable. The predeclared operational gate is:

- exact target context >= 1,000,000 tokens;
- at least three finite validation-loss telemetry observations;
- relative loss span `(max(loss)-min(loss))/mean(loss)` ≤ 5%; and
- retained OOM/crash/restart and distributed-topology receipts.

Passing only establishes this narrow operational loss-drift property; it does
not establish task quality, retrieval correctness, latency, or a win over RAG.
The existing `long_context.py` pilot ends at 1024 tokens and cannot satisfy this
contract.

Memory experiments must record dynamic KV-cache policy, cache hit/retrieval
metrics, LRU eviction, decay function, quantization format/calibration, low-rank
rank/error, peak device memory and throughput. RAG comparisons must lock corpus,
chunking, embedding/retriever/index version, retrieved-token budget, generator,
latency policy and metric direction before either arm is run.

## Theory programme and scope

The required reviewable report needs separate theorem statements and proof
status for: VC/pseudodimension or Rademacher bounds; gradient flow under the
specified low-rank parameterization; saddle-point conditions; information
capacity of sparse memory; addressability versus accessible retrieval; and the
link from effective dimension to generalization. Each theorem must name its
assumptions and explicitly distinguish a reference model from the implemented
hash/cache system.

`paper/ckpt003_theory_scope.tex` is an honest outline and claim ledger, **not**
the requested 20+ page completed proof report. No generalization theorem,
convergence result, information-capacity result, or relationship to the 1M
implementation has yet received independent mathematical review.

## Release gate

`v1.0.0-scale` remains blocked until all four conditions have durable evidence:

1. a completed 1B-parameter training run with a recoverable hashed checkpoint;
2. completed parameter- and FLOP-scaling series with an honestly reported fit;
3. a 1M-token stability receipt under the protocol above; and
4. a theory report ready for independent reviewer reading and its review record.

Independent replication on a separately controlled system is additionally
required before claiming reproducibility at scale.
