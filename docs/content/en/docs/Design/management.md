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

### Multi-stage design

CKKS management in HEIR is easiest to understand as three separate stages:

1. `secret-insert-mgmt-ckks` is structural. It decides where management is
   needed and inserts abstract ops such as `mgmt.modreduce`, `mgmt.relinearize`,
   `mgmt.level_reduce`, and `mgmt.adjust_scale`. At this point the pass is
   deciding *where* scale reconciliation is needed, not yet committing to
   concrete numeric scale targets.
1. `generate-param-ckks` chooses the CKKS scheme parameters.
1. `populate-scale-ckks` resolves the abstract management structure into
   concrete scale annotations, materializes `mgmt.adjust_scale`, and then reruns
   analysis so the achieved scales recorded in the IR remain truthful.
   `secret-to-ckks` lowers this annotated IR to CKKS/LWE ops afterwards.

This split is intentional. It preserves the current management structure used by
mainline HEIR while leaving room for future scale-target policies that need more
information than the structural pass has.

### Current policies

HEIR now has two CKKS target-scale policies:

- `nominal` is the default compatibility policy;
- `precise` is an opt-in policy that uses the real dropped primes $q_i$.

Both policies use the same three-stage pipeline above. The difference is in what
`populate-scale-ckks` does after parameters are known.

#### `nominal`: powers of two, represented exactly

The `nominal` policy deliberately preserves the behavior that HEIR already
supported before high-precision scale tracking. The change is in the
representation, not in the policy:

- fresh CKKS inputs start at the nominal scale $2^{\\mathsf{logDefaultScale}}$;
- with the "before multiplication (including the first multiplication)" policy,
  fresh inputs start at $2^{2 \\cdot \\mathsf{logDefaultScale}}$;
- multiplication multiplies scales exactly;
- rescale/modreduce divides by the nominal power-of-two scale
  $2^{\\mathsf{logDefaultScale}}$, not by the concrete dropped prime $q_i$.

The important upgrade is that these values are stored as exact integers
(`APInt`) representing the scaling factor itself, rather than as exponent-like
`int64_t` placeholders such as `45` or `90`. In other words, the `nominal`
policy is still the old HEIR model, but expressed as exact integer scales like
$2^{45}$ and $2^{90}$.

There is one small legacy exception worth calling out: module-aware CKKS scale
analysis uses the nominal power-of-two policy described here, but a type-only
LWE fallback still approximates its nominal rescale factor from the modulus when
no CKKS scheme parameter attribute is available. That fallback is retained for
compatibility and is not the intended long-term CKKS scale model.

#### `precise`: real-$q_i$ scale evolution with local mgmt refinement

The `precise` policy is the first opt-in mode that makes nontrivial use of those
exact integer scales.

In this mode:

- fresh inputs still start from the configured encode scale;
- multiplication uses exact integer products;
- rescale/modreduce divides by the actual dropped prime $q_i$, rounded to the
  nearest integer;
- `populate-scale-ckks` is allowed to refine the existing mgmt skeleton by
  inserting extra `mgmt.modreduce` and related level adjustments so that
  addition and subtraction sites still see matching exact scales.

This refinement happens *after* parameters are known. `secret-insert-mgmt-ckks`
still decides the structural shape early; `populate-scale-ckks` is the place
where HEIR decides whether the current meeting point is really achievable under
the precise policy and, if not, lowers both sides to a later common meeting
point.

When those extra repairs require more levels than the current modulus chain
provides, `populate-scale-ckks` regenerates a compatible CKKS scheme parameter
attribute with a larger level budget and reruns the precise analysis. The goal
is to keep the precise policy total on arithmetic IR that the historical nominal
pipeline could already lower, even if the precise path needs more management ops
to do so honestly.

### Why `APInt` still matters even before the full BMPH20 design

Moving from exponent-like `int64_t` values to exact integer scale factors is
useful in both policies:

- powers such as $2^{90}$ do not fit in `int64_t`;
- exact multiplication and exact integer ratios are needed when materializing
  `mgmt.adjust_scale` as "multiply by an encoded one";
- the IR records the actual chosen scaling factor rather than an informal
  shorthand for its base-2 logarithm.

The `nominal` policy should be viewed as "exact representation of the existing
nominal model." The `precise` policy goes further by making rescale use the real
$q_i$, but it is still intentionally a conservative first step rather than the
full general BMPH20-style planner.

### Scale-matching invariant and `SameOperandsAndResultType`

The intended invariant remains the same: by the time CKKS addition or
subtraction is emitted, both operands should have the same managed scale, and
the ops can keep `SameOperandsAndResultType`.

This is important enough to say explicitly: high-precision scale tracking does
**not** require weakening `ckks.add` or `ckks.sub`. If a backend wants a
different normalization strategy, that should be expressed as a different
management policy or a different lowering of the management ops, not by making
CKKS add/sub accept mismatched operand types.

Temporary non-add-safe scales are still expected between operations. For
example, a ciphertext can temporarily live at $\\Delta^2$ after multiplication
and before the following rescale. The management pipeline is responsible for
ensuring those transient states do not reach addition or subtraction sites.

### Resolving `mgmt.adjust_scale`

The `secret-insert-mgmt-ckks` pass inserts cross-level adjustment as a sequence
of up to three operations: `mgmt.level_reduce` (drop extra RNS limbs),
`mgmt.adjust_scale` (placeholder for scale correction), and `mgmt.modreduce`
(rescale by one level). Together, these bring a higher-level operand to the same
level and scale as the computation result. See the BGV section above for details
on the cross-level procedure.

In the `nominal` policy, `populate-scale-ckks` resolves `mgmt.adjust_scale`
using the nominal power-of-two model and then materializes it with the existing
generic pattern:

1. compute the desired nominal target scale;
1. compute an exact integer ratio `delta = target / input`;
1. encode the constant one at scale `delta`;
1. replace `mgmt.adjust_scale(x)` with `x * 1_delta`.

In the `precise` policy, the same abstract op is kept longer. First,
`populate-scale-ckks` reruns exact dataflow with the real $q_i$ and checks
whether the current `adjust_scale` plus following rescale chain can land on the
intended exact target. If not, it lowers the meeting point by inserting more
management before the add/sub and tries again. Only after that repair loop
stabilizes does it materialize `mgmt.adjust_scale` using the same generic
"multiply by an encoded one" idea, but with an exact integer `delta` chosen so
that the *post-rescale* result lands on the repaired exact target.

This preserves a useful architectural seam:

- the abstract `mgmt.adjust_scale` op still means "this value needs scale
  reconciliation";
- the policy decides what exact target to ask for;
- the materializer only runs once that target is known to be reachable.

If an `adjust_scale` ends up unconstrained, or if its target scale is already
equal to its input scale, `populate-scale-ckks` erases it as a no-op instead of
forcing a redundant materialization.

### Why this is still not the full BMPH20 / backend-specific design

Standard RNS CKKS does not actually rescale by a symbolic $\\Delta$; it rescales
by the concrete dropped prime $q_i$. Since NTT-friendly primes are not exactly
$2^{\\mathsf{logDefaultScale}}$, real full-RNS CKKS has scale drift that the
nominal power-of-two model intentionally ignores.

That fuller reality now appears in HEIR's optional `precise` policy, but only in
a deliberately local form. The current implementation does **not** yet try to do
global tree-wide scale planning. Instead it takes a smaller and safer step:

- keep the old nominal behavior as the default;
- add a precise mode that repairs the mismatch shapes current mgmt insertion
  already creates;
- preserve strong CKKS IR invariants in both modes;
- and leave room for more global target-scale policies.

### What the current precise mode does and does not do

The current `precise` mode is intentionally useful but still local.

It already gives HEIR two real improvements:

- generic arithmetic IR can be re-repaired with extra management so that exact
  real-$q_i$ scales still meet at add/sub sites;
- on the Orion-to-Lattigo polynomial-evaluation path, HEIR can keep the
  Chebyshev target scale tied to the ciphertext's incoming exact scale instead
  of always resetting it to the nominal default.

This is enough to demonstrate a real benefit from exact post-parameter scale
tracking on the existing BSGS/Paterson-Stockmeyer style path without changing
the polynomial decomposition algorithm itself.

But future policies still matter.

Algorithm 2 of [BMPH20](https://eprint.iacr.org/2020/1203) is the right mental
model for a more advanced global post-parameter policy. It back-propagates
target scales through a polynomial-evaluation tree so that each addition happens
between operands at exactly the same planned scale. Its main benefit is better
precision, and in some cases it also improves level usage. It is not primarily
an operation-count optimization; the paper explicitly allows a small increase in
the number of products in exchange for better scale behavior.

Backend-specific schedules are another natural future policy. For example, a
backend like CHEDDAR may want all ciphertexts at level $i$ to land on one
backend-defined nominal scale for that level, with cross-level adjustment
lowering to a backend-specific "level down" operation instead of the generic
`mul_plain` pattern. That is a policy choice layered on top of the same
management framework, not a reason to relax the generic CKKS IR.

### Example: why richer policies are still needed later

The following synthetic example illustrates why neither the nominal model nor
the current local precise repairs are the last word on CKKS scale planning. Let
$\\Delta = 2^{40}$ and let a real rescale prime be $q_i = \\Delta + 17$. If both
operands of a multiplication start at scale $\\Delta$, then the exact
post-rescale scale is

$$ \\operatorname{round}\\left(\\frac{\\Delta^2}{\\Delta + 17}\\right) = \\Delta
\- 17 $$

Under that real full-RNS behavior, trying to match an operand still at scale
$\\Delta$ using the generic "multiply by an encoded one" trick would require a
plaintext scale of $(\\Delta - 17) / \\Delta$, which is not a positive integer.
So a policy that tries to target the raw physical post-rescale value can run
into targets that the simple generic equalization pattern cannot realize.

This does not mean CKKS addition must accept mismatched scales. It means the
compiler needs a better target-scale policy when it wants to model real CKKS
physics more closely. The current precise mode can repair many such sites by
lowering the meeting point and recomputing an exact reachable `adjust_scale`
delta, but it is still not a global planner. BMPH20-style back-propagation and
backend-specific per-level schedules are two examples of the richer policies
that can be layered on top of the same three-stage management framework.

<!-- mdformat global-off -->
