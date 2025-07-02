---
title: Generate CKKS Parameters
weight: 3
---

## Overview

The `generate-param-ckks` pass automatically generates CKKS (Cheon-Kim-Kim-Song)
homomorphic encryption scheme parameters for approximate arithmetic on real and
complex numbers. Unlike BGV/BFV which focus on exact arithmetic, CKKS is
optimized for floating-point computations with controlled precision loss through
scaling operations.

## Input/Output

- **Input**: MLIR module with `secret` dialect operations annotated with
  `mgmt.mgmt` attributes indicating ciphertext levels and management operations
- **Output**: Same module with `ckks.schemeParam` attribute added at module
  level containing generated parameters (logN, Q, P, logDefaultScale)

## Options

- `--slot-number` (int, default: 0): Minimum number of slots required for
  parameter generation
- `--first-mod-bits` (int, default: 55): Number of bits for the first prime
  coefficient modulus (determines initial precision and security)
- `--scaling-mod-bits` (int, default: 45): Number of bits for the scaling prime
  coefficient modulus (controls precision loss during multiplication/rescaling)
- `--use-public-key` (bool, default: true): Whether to use public key encryption
  (affects security parameters)
- `--encryption-technique-extended` (bool, default: false): Whether to use
  extended encryption technique

## Usage Examples

```bash
# Generate CKKS parameters with default settings
heir-opt --generate-param-ckks input.mlir

# Use custom modulus bit sizes for different precision requirements
heir-opt --generate-param-ckks --first-mod-bits=60 --scaling-mod-bits=40 input.mlir

# Generate parameters for specific slot count
heir-opt --generate-param-ckks --slot-number=8192 input.mlir

# Use secret key encryption only
heir-opt --generate-param-ckks --use-public-key=false input.mlir

# Use extended encryption technique for additional security
heir-opt --generate-param-ckks --encryption-technique-extended=true input.mlir
```

## When to Use

This pass should be used in CKKS-based FHE compilation pipelines after:

1. Circuit has been converted to `secret` dialect operations with floating-point
   types
1. Management operations (`mgmt.relinearize`, `mgmt.modreduce`) have been
   inserted
1. Operations have been annotated with `mgmt.mgmt` attributes (via
   `--annotate-mgmt`)

CKKS is particularly suitable for applications requiring:

- Approximate arithmetic on real/complex numbers
- Machine learning inference with neural networks
- Signal processing and data analytics
- Applications where controlled precision loss is acceptable

## Prerequisites

- Input must contain `mgmt` dialect operations and attributes
- Use `--secret-insert-mgmt-ckks` and `--annotate-mgmt` passes before this pass
- Operations must be properly annotated with ciphertext levels and dimensions
- Typically used with floating-point data types (f16, f32, f64)

## Parameter Considerations

- **First Modulus Bits**: Controls initial precision and security level. Higher
  values provide more precision but require larger parameters
- **Scaling Modulus Bits**: Determines precision loss during rescaling
  operations. Should be balanced with computational depth
- **Modulus Chain**: Generated automatically based on circuit depth and required
  precision levels

## Example Transformation

**Input:**

```mlir
module {
  func.func @add(%arg0: !secret.secret<f16>) -> !secret.secret<f16> {
    %0 = secret.generic(%arg0 : !secret.secret<f16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> !secret.secret<f16>
    return %0 : !secret.secret<f16>
  }
}
```

**Output:**

```mlir
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953], P = [36028797019488257], logDefaultScale = 45>} {
  func.func @add(%arg0: !secret.secret<f16>) -> !secret.secret<f16> {
    %0 = secret.generic(%arg0 : !secret.secret<f16>) attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: f16):
      %1 = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f16
      secret.yield %1 : f16
    } -> !secret.secret<f16>
    return %0 : !secret.secret<f16>
  }
}
```
