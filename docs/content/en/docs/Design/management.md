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

CKKS is a leveled scheme where each level has a modulus $q_l$. The level is
numbered from $0$ to $L$ where $L$ is the input level and $0$ is the output
level. CKKS ciphertext contains a scaled message $\\Delta m$ where $\\Delta$
takes some value like $2^40$ or $2^80$. After multiplication of two messages,
the scaling factor $\\Delta'$ will become larger, hence some kind of management
policy is needed in case it blows up. Contrary to BGV where modulus switching is
used for noise management, in CKKS modulus switching from level $l$ to level
$l-1$ can divide the scaling factor $\\Delta$ by the modulus $q_l$.

HEIR treats CKKS management as one system over:

- ordinary CKKS arithmetic
- opaque structured ops such as `orion.linear_transform` and `orion.chebyshev`
- explicit CKKS expansions of those structured ops

### Common notation

HEIR uses the following notation throughout CKKS management:

- `q_l`: the modulus dropped by an ordinary RNS rescale from level `l` to level
  `l-1`
- `R_l`: the divisor used by a backend when its ordinary rescale at level `l` is
  not division by `q_l`
- `Delta`: one fixed global CKKS scale
- `S_l`: the on-schedule scale at level `l`
- `meet(a, b) = min(level(a), level(b))`: the highest level at which `a` and `b`
  can meet before `add` or `sub`

When ordinary rescale divides by the dropped modulus directly, `R_l = q_l`.

### Backend scale laws

Every CKKS backend presents one ordinary-arithmetic scale law to management.

#### Exact tracked law

There is no per-level scale schedule.

- `add` and `sub` run at `meet(a, b)`
- both operands are brought to one common scale `T` at that level
- the result is `(level, scale) = (meet(a, b), T)`
- `mul` produces `scale_out = scale_a * scale_b`
- rescale from level `l` to level `l-1` produces `scale_out = scale_in / q_l`
- dropping several levels divides by the product of the dropped moduli

This is the ordinary Lattigo CKKS law. Ordinary cross-level addition under this
law means:

1. drop both operands to `meet(a, b)`
1. choose one common scale at that level
1. multiply one operand by the exact integer ratio if the two scales differ

#### Fixed-delta law

One global scale `Delta` is fixed by the parameters.

- every on-schedule ciphertext at every level has scale `Delta`
- ordinary auxiliary plaintexts are encoded at scale `Delta`
- ordinary rescale returns to `Delta`

This is the OpenFHE fixed-scale law. It covers both `FIXEDMANUAL` and
`FIXEDAUTO`: they share the same outer scale schedule even though the library
realizes that schedule differently internally.

#### Q-derived recurrence law

One on-schedule scale `S_l` is attached to each level.

- `S_L` is fixed directly by the parameters
- for `0 < l <= L`, `S_{l-1} = S_l^2 / q_l`
- ordinary rescale from level `l` to level `l-1` returns to `S_{l-1}`

This is the OpenFHE flexible law.

#### Q-derived recurrence law with explicit top pair

This law differs from the previous one only at the top of the chain.

- `S_L` and `S_{L-1}` are fixed directly
- below the top pair, `S_{l-1} = S_l^2 / q_l`

This is the OpenFHE extended flexible law.

#### Rescale-prime-product law

One on-schedule scale `S_l` is attached to each level, but ordinary rescale at
level `l` divides by the backend rescale-prime product `R_l`.

- `S_0` is fixed directly by the backend configuration
- for `0 < l <= L`, `S_{l-1} = S_l^2 / R_l`
- equivalently, `S_l = sqrt(S_{l-1} * R_l)`
- ordinary rescale from level `l` to level `l-1` returns to `S_{l-1}`

This is the CHEDDAR law.

### Structured-op transfer laws

Before coarse management, HEIR chooses one realization style for each Orion op
and records its coarse intrinsic level cost.

- `orion.impl_style`
- `orion.level_cost_ub`

Explicit styles are lowered to ordinary CKKS arithmetic before further CKKS
management. Opaque styles remain in the IR and contribute exact transfer laws to
management.

#### `linear_transform`

An opaque `linear_transform` contributes:

- its intrinsic level cost
- one exact input-to-output transfer law on `(level, scale)`
- any auxiliary plaintext level or auxiliary plaintext scale choices required by
  that law

Representative transfer laws are:

- exact tracked:
  - `out.level = in.level`
  - `out.scale = in.scale * P`
  - `P` is the chosen auxiliary plaintext scale
- fixed-delta:
  - `out.level = in.level`
  - `out.scale = in.scale * Delta`
- q-derived recurrence:
  - `out.level = in.level`
  - `out.scale = in.scale * S_aux`
  - `S_aux` is the on-schedule auxiliary plaintext scale chosen for the
    transform
- rescale-prime-product:
  - `out.level = pt_level - 1`
  - `out.scale = in.scale * P / R_{pt_level}`
  - `pt_level` and `P` are fixed by exact resolution

#### `chebyshev`

An opaque `chebyshev` contributes:

- its intrinsic upper bound on level consumption
- one exact output-scale law at the chosen output level

The output-scale law is one of:

- explicit target-scale:
  - exact resolution chooses the output scale explicitly
- fixed-delta:
  - the output scale is `Delta`
- q-derived recurrence:
  - the output scale is `S_l` at the chosen output level
- q-derived recurrence with explicit top pair:
  - the output scale is `S_l` under that schedule
- rescale-prime-product:
  - the output scale is `S_l` under the backend rescale-prime-product schedule

### Management procedure

Managed CKKS compilation proceeds in five stages:

1. structured-op annotation
1. coarse management insertion
1. parameter selection
1. exact resolution
1. materialization and backend lowering

#### Structured-op annotation

Structured-op annotation chooses the realization style and records the coarse
intrinsic level cost.

- opaque styles stay in the IR and expose exact transfer laws
- explicit styles are lowered to ordinary CKKS arithmetic before further CKKS
  management

#### Coarse management insertion

Coarse insertion chooses the multiply-driven rescale family:

1. after multiplication
1. before multiplication
1. before multiplication, including the first multiplication

It inserts the multiply-side management implied by that choice, including
`mgmt.modreduce`, `mgmt.relinearize`, and any bootstrap placements.

At every secret `add` or `sub`, it inserts `mgmt.reconcile` markers on the
secret operands. A `mgmt.reconcile` marker records only that the merge must be
made level-safe and scale-safe later. It does not choose:

- which side to lower
- the final meeting level
- the final output scale

For each marked merge, coarse insertion records the highest possible meeting
level:

- `highest_meeting_level(add(a, b)) = meet(a, b)`

This is an upper bound, not a final decision.

#### Parameter selection

Parameter selection uses:

- the chosen multiply-driven rescale family
- the coarse multiply-side management skeleton
- the intrinsic level cost of every remaining opaque structured op
- the highest possible meeting level of every marked merge
- the ordinary backend scale law
- the transfer-law family of every remaining opaque structured op

It produces a modulus chain that is large enough for the selected backend scale
law and the selected structured-op realizations.

#### Exact resolution

Exact resolution runs after the parameters are known. At that point HEIR knows:

- the exact ordinary-arithmetic scale law of the selected backend
- the exact per-level schedule implied by that law, when the law is
  schedule-based
- the transfer law of every remaining opaque `linear_transform`
- the output-scale law of every remaining opaque `chebyshev`

Exact resolution assigns a concrete level and a concrete scale to every
remaining value and resolves every `mgmt.reconcile` marker.

For each merge, exact resolution starts from `highest_meeting_level(add(a, b))`
and chooses the final meeting level and final scale plan that are legal under
the active scale law.

- under the fixed-delta law, the legal on-schedule scale at every level is
  `Delta`
- under a schedule law, the legal on-schedule scale at level `m` is `S_m`
- under the exact tracked law, exact resolution chooses the common merge scale
  explicitly

When an opaque `chebyshev` exposes an explicit target-scale law, exact
resolution seeds the backward solve from the required downstream output scale
and applies the op transfer law during that solve.

When an opaque `linear_transform` uses auxiliary plaintexts, exact resolution
also fixes the auxiliary plaintext level and the auxiliary plaintext scale
required by the selected backend law.

For OpenFHE-targeted CKKS pipelines, exact resolution additionally fixes:

- the OpenFHE noise-scale degree required for every encoded plaintext
- the OpenFHE auxiliary plaintext level required by every opaque
  `linear_transform`
- any extra modulus-chain growth required by the selected OpenFHE scale law
  before lowering to the OpenFHE dialect

#### Materialization and backend lowering

After exact resolution:

- every CKKS value has an exact level and exact scale
- every `add` and `sub` is level-safe and scale-safe
- every remaining opaque structured op has fully instantiated level and scale
  parameters under its transfer law

At that point `mgmt.adjust_scale`, `mgmt.level_reduce`, and `mgmt.modreduce` are
fully materialized, and only then does HEIR lower to backend-specific scheme
ops.

For CKKS, backend crypto-context configuration is a readout stage:

- it reads the chosen modulus chain from `ckks.schemeParam`
- it reads the selected backend scale law
- it reads the feature set that determines which evaluation keys to generate
- it does not choose a new CKKS modulus chain
- it does not redo CKKS level or scale analysis

### Scale-matching invariant

The CKKS IR remains strict:

- by the time `ckks.add` or `ckks.sub` is emitted, both operands have the same
  level and the same managed scale
- `ckks.add` and `ckks.sub` keep `SameOperandsAndResultType`

Temporary off-schedule scales are allowed between operations. Exact resolution
eliminates them before any addition or subtraction survives to the CKKS IR.

<!-- mdformat global-off -->
