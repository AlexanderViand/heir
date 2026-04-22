---
title: Ciphertext Management
weight: 9
---

On 2025-04-17, Hongren Zheng gave a talk overview of the ciphertext management
system in the HEIR working group meeting.
[The video can be found here](https://youtu.be/HHU6rCMxZRc?si=U_ePY5emqs6e4NoV&t=1631)
and [the slides can be found here](/slides/mgmt-2025-04-17.pdf)

## Introduction

To lower from user specified computation to FHE scheme operations, a compiler
must insert *ciphertext management* operations to satisfy various requirements
of the FHE scheme, like modulus switching, relinearization, and bootstrapping.
In HEIR, such operations are modeled in a scheme-agnostic way in the `mgmt`
dialect.

Taking the arithmetic pipeline as example: a program specified in high-level
MLIR dialects like `arith` and `linalg` is first transformed to an IR with only
`arith.addi/addf`, `arith.muli/mulf`, and `tensor_ext.rotate` operations. We
call this form the *secret arithmetic* IR.

Then management passes insert `mgmt` ops to support future lowerings to scheme
dialects like `bgv` and `ckks`. As different schemes have different management
requirement, they should be inserted in different styles.

We discuss each scheme below to show the design in HEIR. For RLWE schemes, we
all assume RNS instantiation.

## BGV

BGV is a leveled scheme where each level has a modulus $q_i$. The level is
numbered from $0$ to $L$ where $L$ is the input level and $0$ is the output
level. The core feature of BGV is that when the magnititude of the noise becomes
large (often caused by multiplication), a modulus switching operation from level
$i$ to level $i-1$ can be inserted to reduce the noise to a "constant" level. In
this way, BGV can support a circuit of multiplicative depth $L$.

### BGV: Relinearization

HEIR initially inserts relinearization ops immediately after each multiplication
to keep ciphertext dimension "linear". A later relinearization optimization pass
relaxes this requirement, and uses an integer linear program to decide when to
relinearize. See [Optimizing Relinearization](/docs/design/relinearization_ilp/)
for more details.

### BGV: Modulus switching

There are several techniques to insert modulus switching ops.

For the example circuit `input -> mult -> mult -> output`, the insertion result
could be one of

1. After multiplication: `input -> (mult -> ms) -> (mult -> ms) -> output`

1. Before multiplication: `input -> (mult) -> (ms -> mult) -> (ms -> output)`

1. Before multiplication (including the first multiplication):
   `input -> (ms -> mult) -> (ms -> mult) -> (ms -> output)`

The first strategy is from the BGV paper, the second and third strategies are
from OpenFHE, which correspond to the `FLEXIBLEAUTO` mode and `FLEXIBLEAUTOEXT`
mode, respectively.

The first strategy is conceptually simpler, yet other policies have the
advantage of smaller noise growth. In latter policies, by delaying the modulus
switch until just before multiplication, the noise from other operations between
multiplications (like rotation/relinearization) also benefit from the noise
reduction of a modulus switch.

Note that, as multiplication has two operands, the actual circuit for the latter
two policies is `mult(ms(ct0), ms(ct1))`, whereas in the first policy the
circuit is `ms(mult(ct0, ct1))`.

The third policy has one more switching op than the others, so it will need one
more modulus.

There are also other insertion strategy like inserting it dynamically based on
current noise (see HElib) or lazy modulus switching. Those are not implemented.

### BGV: Scale management

For the original BGV scheme, it is required to have $qi \\equiv 1 \\pmod{t}$
where $t$ is the plaintext modulus. However in practice such requirement will
make the choice of $q_i$ too constrained. In the GHS variant, this condition is
removed, with the price of scale management needed.

Modulus switching from level $i$ to level $i-1$ is essentially dividing (with
rounding) the ciphertext by $q_i$, hence dividing the noise and payload message
inside by $q_i$. The message $m$ can often be written as $\[m\]\_t$, the coset
representative of `m` $\\mathbb{Z}/t\\mathbb{Z}$. Then by dividing of $q_i$
produces a result message $\[m \\cdot q_i^{-1}\]\_t$.

Note that when $qi \\equiv 1 \\pmod{t}$, the result message is the same as the
original message. However, in the GHS variant this does not always hold, so we
call the introduced factor of $\[q^{-1}\]\_t$ the *scale* of the message. HEIR
needs to record and manage it during compilation. When decrypting the scale must
be removed to obtain the right message.

Note that, for messages $m_0$ and $m_1$ of different scale $a$ and $b$, we
cannot add them directly because $\[a \\cdot m_0 + b \\cdot m_1\]\_t$ does not
always equal $\[m_0 + m_1\]\_t$. Instead we need to adjust the scale of one
message to match the other, so $\[b \\cdot m_0 + b \\cdot m_1\]\_t = \[b \\cdot
(m_0 + m_1)\]\_t$. Such adjustment could be done by multiplying $m_0$ with a
constant $\[b \\cdot a^{-1}\]\_t$. This adjustment is not for free, and
increases the ciphertext noise.

As one may expect, different modulus switching insertion strategies affect
message scale differently. For $m_0$ with scale $a$ and $m_1$ with scale $b$,
the result scale would be

1. After multiplication: $\[ab / qi\]\_t$.

1. Before multiplication: $\[a / qi \\cdot b / qi\]\_t = \[ab / (qi^2)\]\_t$.

This is messy enough. To ease the burden, we can impose additional requirement:
mandate a constant scale $\\Delta_i$ for all ciphertext at level $i$. This is
called the *level-specific scaling factor*. With this in mind, addition within
one level can happen without caring about the scale.

1. After multiplication: $\\Delta\_{i-1} = \[\\Delta_i^2 / qi\]\_t$

1. Before multiplication: $\\Delta\_{i-1} = \[\\Delta_i^2 / (qi^2)\]\_t$

### BGV: Cross Level Operation

With the level-specific scaling factor, one may wonder how to perform addition
and multiplication of ciphertexts on different levels. This can be done by
adjusting the level and scale of the ciphertext at the higher level.

The level can be easily adjusted by dropping the extra limbs, and scale can be
adjusted by multiplying a constant, but because multiplying a constant will
incur additional noise, the procedure becomes the following:

1. Assume the level and scale of two ciphertexts are $l_1$ and $l_2$, $s_1$ and
   $s_2$ respectively. WLOG assume $l_1 > l_2$.

1. Drop $l_1 - l_2 - 1$ limbs for the first ciphertext to make it at level $l_2
   \+ 1$, if those extra limbs exist.

1. Adjust scale from $s_1$ to $s_2 \\cdot q\_{l_2 + 1}$ by multiplying $\[s_2
   \\cdot q\_{l_2 + 1} / s1\]\_t$ for the first ciphertext.

1. Modulus switch from $l_2 + 1$ to $l_2$, producing scale $s_2$ for the first
   ciphertext and its noise is controlled.

### BGV: Implementation in HEIR

In HEIR the different modulus switching policy is controlled by the pass option
for `--secret-insert-mgmt-bgv`. The pass defaults to the "Before Multiplication"
policy. If user wants other policy, the `after-mul` or
`before-mul-include-first-mul` option may be used. The `mlir-to-bgv` pipeline
option `modulus-switch-before-first-mul` corresponds to the latter option.

The `secret-insert-mgmt` pass is also responsible for managing cross-level
operations. However, as the scheme parameters are not generated at this point,
the concrete scale could not be instantiated so some placeholder operations are
inserted.

After the modulus switching policy is applied, the `generate-param-bgv` pass
generates scheme parameters. Optionally, user could skip this pass by manually
providing scheme parameter as an attribute at module level.

Then `populate-scale-bgv` comes into play by using the scheme parameters to
instantiate concrete scale, and turn those placeholder operations into concrete
multiplication operation.

## CKKS

CKKS is a leveled scheme where each level has a modulus $q_i$. The level is
numbered from $0$ to $L$ where $L$ is the input level and $0$ is the output
level. CKKS ciphertext contains a scaled message $\\Delta m$ where $\\Delta$
takes some value like $2^40$ or $2^80$. After multiplication of two messages,
the scaling factor $\\Delta'$ will become larger, hence some kind of management
policy is needed in case it blows up. Contrary to BGV where modulus switching is
used for noise management, in CKKS modulus switching from level $i$ to level
$i-1$ can divide the scaling factor $\\Delta$ by the modulus $q_i$.

The management of CKKS is similar to BGV above in the sense that their strategy
are the similar and uses similar code base. However, BGV scale management is
internal and users are not required to concern about it, while CKKS scale
management is visible to user as it affects the precision. One notable
difference is that, for "Before multiplication (including the first
multiplication)" modulus switching policy, the user input should be encoded at
$\\Delta^2$ or higher, as otherwise the first modulus switching (or rescaling in
CKKS term) will rescale $\\Delta$ to $1$, rendering full precision loss.

### Current CKKS pipeline

The old "insert management, generate parameters, populate scales" story is now
too coarse to describe the CKKS implementation accurately. The current CKKS path
is better understood as the following sequence:

1. `annotate-orion` selects an implementation style for high-level Orion ops and
   attaches a coarse intrinsic level-cost upper bound to each of them.
1. `secret-insert-mgmt-ckks` performs a coarse forward structural pass. It still
   inserts ordinary multiply-driven management such as `mgmt.modreduce` and
   `mgmt.relinearize`. Secret `add`/`sub` sites receive `mgmt.reconcile` markers
   rather than an early hard-coded repair sequence, although multiply
   preparation is still handled eagerly where needed so that `mul` remains
   well-formed.
1. `generate-param-ckks` chooses a first candidate modulus chain from that
   coarse managed IR.
1. `resolve-reconcile-ckks` resolves coarse reconciliation sites for the local
   and canonical-per-level policies, and the current q-aware path still runs it
   first as a preliminary local reconciliation step.
1. `resolve-scale-ckks-bmph20` is the current q-aware iterative refinement pass.
   It starts from the candidate parameters and previously resolved local
   reconciliation shape, then may refine scales and grow the modulus chain if
   the q-aware plan needs more levels.
1. Exact scale analysis then annotates the achieved scales and materializes any
   remaining `mgmt.adjust_scale` placeholders.

`secret-to-ckks` lowers this resolved IR to CKKS/LWE ops afterwards, and
`scheme-to-<backend>` lowerings are expected to honor the implementation-style
contracts attached to any remaining high-level Orion ops.

### Implementation-style annotation for high-level ops

High-level ops such as `orion.linear_transform` and `orion.chebyshev` now carry
two generic, policy-independent annotations:

- `orion.impl_style`, naming the requested implementation family;
- `orion.level_cost_ub`, a coarse intrinsic upper bound on level consumption.

These annotations are selected by `annotate-orion`. They are intentionally not
backend-named. Instead, later lowerings such as `scheme-to-lattigo` or
`scheme-to-cheddar` must either:

- recognize and honor the requested implementation style; or
- fail loudly if that style is not available for that backend.

Today the built-in styles are:

- `orion.linear_transform`: `diagonal-bsgs`, with coarse level cost `0`;
- `orion.chebyshev`: `bsgs`, with coarse level cost $\\lceil
  \\log_2(\\mathsf{degree}) \\rceil$.

The important design point is that these are *intrinsic* coarse facts about the
chosen high-level realization. They are not themselves reconciliation or
target-scale policies.

### Coarse `insert-mgmt`: mul structure plus reconciliation constraints

`secret-insert-mgmt-ckks` is now deliberately less opinionated about
addition/subtraction repair.

It still chooses the broad multiply-driven modulus-switching family:

- `after-mul`;
- `before-mul`;
- `before-mul-include-first-mul`.

It also still inserts the ordinary multiply-side management implied by that
choice, such as `mgmt.modreduce` and `mgmt.relinearize`.

But at secret `add` and `sub` sites it no longer commits early to one specific
cross-level repair sequence. Instead it inserts `mgmt.reconcile` markers on the
secret operands of the binary op. Those markers mean only:

- this merge site will need level/scale reconciliation later;
- the coarse forward pass has *not yet* decided which side to lower, what exact
  meeting level to use, or what exact scale target to request.

This makes the coarse pass simpler and keeps the early IR compatible with both
local canonical policies and later q-aware refinement.

### Candidate parameter generation

`generate-param-ckks` still runs on the coarse managed IR.

At this stage HEIR knows:

- the chosen multiply/rescale family;
- the coarse management skeleton inserted around multiplies;
- the coarse intrinsic level cost of any high-level Orion ops;
- that each secret `add`/`sub` may need later reconciliation.

That is enough to produce a first candidate modulus chain. For the local
power-of-two and canonical-per-level policies, this candidate parameter set is
normally final. For the q-aware iterative policy described below, it is only a
starting point.

### Reconciliation policies after parameters are known

Once parameters are known, HEIR resolves `mgmt.reconcile` markers according to
an explicit reconciliation policy.

#### `local-highest-meeting-point`

This is the generic local policy. At each merge site it tries to reconcile the
operands at the highest available meeting level and materializes the familiar
sequence built from:

- `mgmt.level_reduce`;
- `mgmt.adjust_scale`;
- `mgmt.modreduce`.

This is the best match for HEIR's historical power-of-two behavior.

#### `canonical-per-level`

This is the policy currently used by CHEDDAR-oriented examples. It still
resolves each merge locally, but it assumes that values should be brought onto a
canonical per-level schedule instead of immediately forcing the generic
`adjust_scale + modreduce` shape. In practice this produces the right symbolic
shape for later CHEDDAR lowering, where `scheme-to-cheddar` can realize the
chosen per-level schedule with backend-specific operations such as level-down.

The important distinction is that both policies use the same coarse
`mgmt.reconcile` markers. They differ only in the *post-parameter resolution*
step.

### Exact scale population for the power-of-two compatibility policy

After reconciliation markers are resolved, `populate-scale-ckks` performs exact
integer scale analysis and materialization.

The default CKKS scale policy remains `nominal`. It preserves HEIR's historical
power-of-two scale model:

- fresh inputs start at $2^{\\mathsf{logDefaultScale}}$;
- with `before-mul-include-first-mul`, fresh inputs start at $2^{2 \\cdot
  \\mathsf{logDefaultScale}}$;
- multiplication multiplies scales exactly;
- rescale/modreduce divides by the nominal power-of-two
  $2^{\\mathsf{logDefaultScale}}$, not by the concrete dropped prime $q_i$.

The upgrade is that these scales are now stored as exact integer factors
(`APInt`) such as $2^{45}$ or $2^{90}$ rather than as exponent-like placeholders
such as `45` or `90`.

`populate-scale-ckks` then resolves `mgmt.adjust_scale` by computing an exact
integer plaintext scale `delta`, encoding the constant one at that scale, and
replacing the placeholder with a `mul_plain`-style pattern. If an adjustment is
already satisfied, it is erased as a no-op.

For backwards compatibility, standalone `populate-scale-ckks` will also resolve
any remaining `mgmt.reconcile` markers first if the caller did not run
`resolve-reconcile-ckks` explicitly.

There is one small legacy exception: module-aware CKKS scale analysis uses the
nominal power-of-two policy described here, but a type-only LWE fallback still
approximates its rescale factor from the modulus when no CKKS scheme parameter
attribute is available.

### Q-aware iterative mode and the current BMPH20-shaped pass

HEIR also has an opt-in q-aware mode, exposed today through
`resolve-scale-ckks-bmph20`.

This mode:

- keeps the same coarse forward `insert-mgmt` structure;
- starts from the candidate parameters chosen by `generate-param-ckks`;
- runs q-aware scale analysis where modreduce divides by the actual dropped
  prime $q_i$, rounded to the nearest integer;
- uses a bounded repair loop to refine the management structure when the
  original local meeting point is not exactly reachable;
- and can regenerate a larger CKKS modulus chain if the refined plan requires
  more levels than the current candidate parameters provide.

The current implementation is intentionally narrower than the full
[BMPH20](https://eprint.iacr.org/2020/1203) Algorithm 2. It is best understood
as q-aware forward scale evolution plus iterative local repair, not yet as a
global tree-wide target-scale solver. In particular:

- the repair loop currently rewrites existing `mgmt.adjust_scale` sites and
  lowers meeting points when necessary;
- it is bounded and fails hard if it does not converge within the configured
  iteration cap;
- it is the one policy in the current tree that may invalidate the first
  candidate parameter set and ask the CKKS parameter construction logic for a
  longer modulus chain.

This is why candidate parameters are only provisional for the q-aware path.
Unlike the local power-of-two and canonical-per-level policies, the q-aware
planner may discover that an honest exact realization needs extra level drops.

There is one important backend caveat today: CHEDDAR deliberately does **not**
use the generic q-aware exact rescale model yet. Requesting `ckks-scale-policy`
`precise` on a CHEDDAR module keeps the symbolic choice recorded on the module,
but the active exact scale model falls back to the nominal one there until
CHEDDAR grows its own dedicated q-aware or canonical-schedule realization.

### Scale-matching invariant and `SameOperandsAndResultType`

The core CKKS invariant does not change:

- by the time CKKS addition or subtraction is emitted, both operands should have
  the same managed scale and level;
- `ckks.add` and `ckks.sub` keep `SameOperandsAndResultType`.

Temporary non-add-safe scales are still expected between operations. For
example, a ciphertext can temporarily live at $\\Delta^2$ after multiplication
and before the following rescale. The management pipeline is responsible for
ensuring those transient states do not reach addition or subtraction sites.

This is why HEIR models more sophisticated behavior with management policy and
resolution passes rather than by weakening the generic CKKS IR.

### Why richer policies still matter later

The local power-of-two and canonical-per-level policies are stable because each
level has an obvious target family:

- the nominal power-of-two bucket for that level;
- or the backend-defined canonical scale for that level.

The q-aware/BMPH20-shaped path is harder because real full-RNS CKKS does not
rescale by a symbolic $\\Delta$; it rescales by the concrete dropped prime
$q_i$. Since NTT-friendly primes are not exactly
$2^{\\mathsf{logDefaultScale}}$, real full-RNS CKKS has scale drift that the
nominal model intentionally ignores.

That is why Algorithm 2 of BMPH20 remains the right long-term mental model for
structured polynomial evaluation: it back-propagates target scales through the
evaluation tree so that additions are correct by construction. Its main benefit
is better precision, and in some cases it also improves level usage. It is not
primarily an operation-count optimization.

The current q-aware iterative pass is a useful intermediate step. It
demonstrates that HEIR can already:

- keep strict CKKS IR invariants;
- use exact integer scales and real dropped primes when desired;
- refine management locally when the coarse plan is not physically reachable;
- and preserve compatibility with arithmetic IR that the historical nominal
  pipeline could already lower.

It is not yet the final global planner for all CKKS policies, but it is now a
separate stage with a cleaner contract than the old "everything interesting
happens inside populate-scale" design.

### Backend capability notes for structured CKKS ops

For high-level ops such as `orion.linear_transform` and `orion.chebyshev`, the
interesting backend differences are not captured well by high-level algorithm
names alone. Two implementations may both be described as "BSGS" or
"Paterson-Stockmeyer" while still exposing very different contracts to the
compiler.

#### Linear transform

The intrinsic level cost of a linear transform remains `0`: it is built from
rotations, plaintext multiplication, and addition. The important differences are
instead about *reuse* and *what the backend API lets the compiler express*.

- Lattigo exposes a reusable linear-transform evaluator that can evaluate many
  transforms on the same ciphertext while sharing one decomposition of the input
  and a cache of pre-rotated ciphertexts.
- OpenFHE has a strong native single-transform implementation that already uses
  a two-stage hoisted baby-step/giant-step algorithm internally, but it does not
  expose the same many-transform reusable contract directly.
- CHEDDAR has a native hoisting-based linear-transform object with persistent
  hoist state, but again not the same "many sibling transforms on one input"
  contract that Lattigo exposes.

This distinction matters for larger models more than for small MLP examples. A
plain MLP often has one linear transform per activation, so Lattigo's stronger
reuse API is only weakly exercised. The difference becomes much more important
for workloads that apply multiple related transforms to the same ciphertext, for
example:

- multi-block linear layers;
- attention-style Q/K/V projections;
- any pipeline stage that fans one activation out into many sibling transforms.

In those cases, the Lattigo contract can reduce repeated decomposition and
rotation work. That is primarily a *runtime* advantage. It does not by itself
reduce intrinsic level consumption, and there is no strong reason to expect a
large inherent *noise* advantage from the linear-transform API alone.

#### Polynomial evaluation and Chebyshev

For polynomial evaluation, the important differences are about *who controls the
target scale*, *who owns basis/domain conversion*, and *what precomputation is
reusable*.

- Lattigo's native evaluator is target-scale-driven. The compiler supplies a
  desired target scale, and the evaluator performs a structured decomposition
  against that target.
- OpenFHE exposes a rich native menu of power-basis and Chebyshev-series
  evaluators, including precomputation objects for reusable powers/polynomials,
  but much of the level/scale reconciliation remains internal to the library
  algorithm itself.
- CHEDDAR's native polynomial evaluation path behaves like a compiled
  evaluation-tree engine with a canonical per-level schedule: it is given an
  input level/scale contract and a target scale, then compiles and evaluates
  against that schedule.

This is why Lattigo is currently the best fit for HEIR's q-aware/BMPH20-shaped
planner. The main expected benefit is *accuracy* and *scale discipline*, not
fewer arithmetic operations. OpenFHE can still support strong native polynomial
evaluation, but its contract is less "compiler chooses the exact target scale"
and more "compiler chooses the evaluation family and lets the library manage
more of the internal scale/level details."

For polynomial evaluation, the likely benefits of richer HEIR management are:

- better accuracy or more predictable achieved scale;
- in some cases, better level usage;
- not necessarily a large reduction in operation count relative to native
  structured evaluators that are already asymptotically good.

#### Why current implementation-style names are still too coarse

The current built-in styles:

- `orion.linear_transform = diagonal-bsgs`
- `orion.chebyshev = bsgs`

are therefore only placeholders. They are good enough to select one currently
implemented lowering family, but they are not yet precise enough to describe the
real implementation contracts above.

In particular, future implementation styles will need to distinguish details
such as:

- one-shot versus reusable-many linear transforms;
- whether hoisting state can be shared across many sibling transforms;
- explicit target-scale control versus library-managed scale reconciliation;
- explicit HEIR CKKS expansion versus opaque native backend evaluator calls.

### Roadmap toward cross-backend parity

The long-term goal is not merely to make every program lower to every backend.
The goal is to make structured models such as MLPs, and later larger Orion
models, run on Lattigo, OpenFHE, and CHEDDAR with nearly the same algorithmic
quality, accuracy, and performance whenever the underlying backend APIs make
that possible.

The cleanup plan is therefore:

1. Keep the current coarse-management architecture.

   - `annotate-orion` chooses an implementation style and coarse intrinsic level
     cost.
   - coarse `secret-insert-mgmt-ckks` inserts multiply-side management and
     `mgmt.reconcile` constraints.
   - candidate parameters are generated from that coarse IR.

1. Split the post-parameter world cleanly by policy.

   - the local power-of-two compatibility path remains the default for generic
     arithmetic and OpenFHE-style nominal lowering;
   - the canonical-per-level path remains the natural fit for CHEDDAR;
   - the q-aware/BMPH20-shaped path becomes the preferred precise path for
     structured polynomial evaluation on backends that can honor explicit target
     scales well, especially Lattigo.

1. Grow implementation-style annotations into real backend contracts.

   - The current placeholder style names must be refined so they encode the
     actual reusable contract that later lowering expects, not just a loose
     algorithm name.
   - Later `scheme-to-<backend>` passes must keep treating those style
     annotations as hard contracts and fail loudly when the requested contract
     is unavailable.

1. Add explicit OpenFHE manual-mode targeting.

   - OpenFHE's automatic scaling modes can silently repair mistakes and hide
     compiler-management bugs.
   - A strict manual-mode path is therefore important both for test confidence
     and for understanding how far HEIR's own management can go without backend
     rescue logic.

1. Support both native and explicit-CKKS realizations of structured ops.

   - When a backend has a strong native implementation that matches the
     requested style contract, HEIR should be able to use it.
   - When that contract is unavailable, HEIR should instead lower the op to
     explicit CKKS operations, or to a slightly richer CKKS layer if new
     scheme-mechanical ops are needed to capture reusable patterns such as
     hoisting.

1. Expand tests around both policy and backend dimensions.

   - small focused tests for reconcile policy differences;
   - focused tests for implementation-style mismatch failures;
   - structured-op tests that exercise native versus explicit realizations;
   - end-to-end tests for the same model under multiple management policies and
     backends;
   - OpenFHE manual-mode end-to-end tests to make sure HEIR is not relying on
     hidden automatic repair.

1. Use larger structured examples as capability tests.

   - Small MLPs are useful smoke tests but are often too simple to expose the
     real value of reusable-many linear-transform contracts.
   - Larger blocked layers, attention-like shapes, and deeper polynomial-heavy
     models are better capability tests for deciding whether two backends are
     truly on par for a requested implementation style.

<!-- mdformat global-off -->
