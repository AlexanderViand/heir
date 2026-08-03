# Global bootstrap placement for HEIR CKKS (orion-style level-DAG port)

## Why

HEIR's `SecretInsertMgmtCKKS` places bootstraps with a greedy waterline
(`BootstrapWaterLine` in `SecretInsertMgmtPatterns.cpp`), which carries
`TODO(#1642): make it work with cross-level operation`. Deep composite-sign
ReLUs (`x*step(x/B)`, sign = 3 chained minimax Chebyshev polys, depth ~14) hit
this gap: the matching pushes the `x*step` multiply to the bottom of the modulus
chain, where its mandatory post-mul rescale overflows → a malformed
`ckks.rescale` onto the full chain that fails verification.

A greedy detect-and-bootstrap loop does NOT converge: HEIR's `LevelState` uses
the same `isMaxLevel`/`Invalid` sentinels for structural artifacts (every
`level_reduce_min` from cross-level matching) and for genuine exhaustion, so
concrete-only detection misses the real operand and sentinel detection explodes
(observed 5262 bootstraps). The robust fix is a GLOBAL placement that works with
concrete level numbers and a known budget `l_eff`, like orion's BootstrapSolver
(orion/core/{level_dag,auto_bootstrap}.py).

## orion's algorithm (reference)

- Network = DAG of modules; each module has a multiplicative `depth`.
- LevelDAG: each module `n` becomes nodes `n@l=0 .. n@l=l_eff`. Node weight =
  layer latency at that level (inf if `level < depth`). Edge
  `prev@l=i -> curr@l=j` weight = bootstrap cost: 0 if `j <= i - prev.depth` (no
  bootstrap), inf if `i - prev.depth <= 0` (prev can't even run), else `t_boot`
  (bootstrap).
- Fork-join (residual) regions are collapsed innermost-first into a single
  aggregate edge per (fork_level, join_level) pair = sum of shortest paths
  through each branch (`LevelDAG.__add__`), reducing the DAG to a path.
- One topological-order shortest-path (`shortest_path`) over the whole thing
  gives the optimal level per module + the input level; bootstraps are where a
  level-increase edge was taken.

## HEIR port plan (phased)

Granularity: OP level inside `secret.generic` (HEIR has no module abstraction).

- Node = each secret-typed SSA value (or op result). `depth(op)` = 1 for ct*ct
  `arith.mulf`/`muli` and ct*pt mul (post-mul rescale), 0 for
  add/sub/rotate/etc.
- `l_eff` = usable levels per bootstrap cycle (from the scheme; ~chain length).
- Joins: an op with >=2 secret operands forces its operands to a common level.
  Plain shortest-path can't enforce this -> need SESE decomposition like orion
  (collapse each single-entry/single-exit region between a fork value and its
  reconvergence into an aggregate edge over (entryLevel, exitLevel) pairs).

Phasing:

1. Build the op-level DAG + depth model over secret.generic; compute SESE
   regions (fork = value with >1 use; join = its reconvergence / the op that
   consumes the merged result). Use a dominator/post-dominator analysis on the
   value graph.
1. LevelDAG construction: per-op level-variants, edge weights per orion's
   bootstrap rule (using concrete levels, l_eff). Aggregate SESE regions.
1. Topological shortest-path -> per-op level assignment + input level.
1. Materialize: insert `mgmt.bootstrap` on level-increase transitions; set each
   op's level (so SecretToCKKS/param-gen size the chain to the solved budget);
   feed into existing scale management (AnnotateMgmt / PopulateScaleCKKS).
1. Replace `insertBootstrapWaterLine` (gate behind an option so the greedy path
   stays default for non-composite models until validated).

Integration point: a new function in `lib/Transforms/SecretInsertMgmt/` (e.g.
`solveBootstrapPlacement`) called from `runInsertMgmtPipeline` instead of
`insertBootstrapWaterLine` when an `--optimize-bootstrap-placement` option is
set; thread that option up through MlirToRLWEPipelineOptions /
torch-linalg-to-ckks like `use-composite-relu`.

Validation: ToyHELRM composite must (a) pass CKKS verification, (b) match
cleartext 0.004490 within ~10%. Then CriteoHELRM. Watch bootstrap count stays
sane (single-digit-ish per ReLU, not the 5262 explosion).

See \[\[project-composite-relu-level-mgmt\]\] for the full diagnosis trail.

<!-- mdformat global-off -->
