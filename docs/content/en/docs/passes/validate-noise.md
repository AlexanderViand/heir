---
title: Validate Noise
weight: 18
---

## Overview

The `validate-noise` pass validates FHE circuits against specified noise models
to ensure they will decrypt correctly with the given parameters. This pass
performs noise analysis throughout the computation and reports whether the
accumulated noise stays within acceptable bounds for successful decryption.

## Input/Output

- **Input**: MLIR module with FHE scheme parameters annotated (typically by
  `generate-param-*` passes)
- **Output**: Same module (unchanged) with noise validation results reported via
  debug output

## Options

- `--model` (string, default: "bgv-noise-kpz21"): Noise model to use for
  validation
  - Must match the model used for parameter generation
  - Available models depend on the FHE scheme (BGV, BFV, CKKS)
- `--annotate-noise-bound` (bool, default: false): Add noise bound annotations
  to the IR
  - Useful for debugging and further analysis
  - Adds attributes to operations with computed noise levels

## Usage Examples

```bash
# Validate with default BGV noise model
heir-opt --validate-noise input.mlir

# Use specific noise model matching parameter generation
heir-opt --validate-noise --model=bgv-noise-mp24 input.mlir

# Validate and annotate noise bounds in IR
heir-opt --validate-noise --annotate-noise-bound=true input.mlir

# View detailed noise analysis (requires debug build)
heir-opt --debug-only=ValidateNoise --validate-noise input.mlir
```

## When to Use

This pass should be used to verify FHE circuits:

1. **After Parameter Generation**: Validate that generated parameters are
   sufficient
1. **During Development**: Ensure circuit modifications don't exceed noise
   budgets
1. **Before Deployment**: Final validation that circuits will work correctly
1. **For Debugging**: Identify where noise bounds are violated

Typical pipeline position:

```bash
heir-opt --generate-param-bgv --validate-noise input.mlir
```

## Noise Models

The available noise models correspond to those used in parameter generation:

### BGV Models

- `bgv-noise-kpz21` or `bgv-noise-by-bound-coeff-average-case`
- `bgv-noise-by-bound-coeff-worst-case`
- `bgv-noise-mp24` or `bgv-noise-by-variance-coeff`
- `bgv-noise-mono`

### BFV Models

- `bfv-noise-kpz21` or `bfv-noise-by-bound-coeff-worst-case`
- `bfv-noise-by-bound-coeff-average-case`
- `bfv-noise-bmcm23` or `bfv-noise-by-variance-coeff`

### CKKS Models

- CKKS uses simpler noise analysis (typically no model selection needed)

## Output Analysis

### Debug Output Format

When using `--debug-only=ValidateNoise`, the output shows:

```bash
Noise Bound: 29.27 Budget: 149.73 Total: 179.00 for value: <block argument> of type 'tensor<8xi16>' at index: 0
Noise Bound: 35.42 Budget: 144.58 Total: 180.00 for value: %0 = arith.addi ... : tensor<8xi16>
```

### Interpretation

- **Noise Bound**: Actual noise level in the ciphertext
- **Budget**: Remaining noise budget before decryption failure
- **Total**: Total available noise budget (Bound + Budget)
- **Status**: Pass/Fail indication for each value

### Validation Results

- **Success**: All operations stay within noise bounds
- **Failure**: At least one operation exceeds acceptable noise levels
- **Warnings**: Operations approaching noise limits

## Noise Budget Management

### Understanding Noise Growth

Different operations contribute differently to noise:

- **Addition**: Minimal noise growth
- **Multiplication**: Significant noise increase
- **Relinearization**: Adds noise but reduces ciphertext size
- **Rescaling (CKKS)**: Reduces noise but loses precision

### Common Issues

- **Insufficient Parameters**: Circuit requires larger parameters
- **Suboptimal Relinearization**: Too many or poorly placed relinearizations
- **Deep Circuits**: Multiplication depth exceeds parameter capabilities

## Example Validation

**Input Circuit:**

```mlir
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [4294991873], P = [4295049217], plaintextModulus = 65537>} {
  func.func @compute(%arg0: !secret.secret<i16>, %arg1: !secret.secret<i16>) -> !secret.secret<i16> {
    %0 = secret.generic(%arg0, %arg1 : !secret.secret<i16>, !secret.secret<i16>)
         attrs = {arg0 = {mgmt.mgmt = #mgmt.mgmt<level = 0>},
                  arg1 = {mgmt.mgmt = #mgmt.mgmt<level = 0>}} {
    ^body(%input0: i16, %input1: i16):
      %1 = arith.addi %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
      %2 = arith.muli %1, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : i16
      secret.yield %2 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }
}
```

**Debug Output:**

```bash
Noise Bound: 12.34 Budget: 166.66 Total: 179.00 for value: %arg0 (fresh ciphertext)
Noise Bound: 12.34 Budget: 166.66 Total: 179.00 for value: %arg1 (fresh ciphertext)
Noise Bound: 12.85 Budget: 166.15 Total: 179.00 for value: %1 = arith.addi (after addition)
Noise Bound: 45.23 Budget: 133.77 Total: 179.00 for value: %2 = arith.muli (after multiplication)
VALIDATION: PASS - All operations within noise bounds
```

## Troubleshooting Noise Violations

When validation fails:

### 1. Increase Parameters

```bash
# Generate larger parameters
heir-opt --generate-param-bgv --slot-number=8192 input.mlir
```

### 2. Optimize Relinearization

```bash
# Optimize placement to reduce noise
heir-opt --optimize-relinearization --validate-noise input.mlir
```

### 3. Use Better Noise Models

```bash
# Try more accurate noise models
heir-opt --generate-param-bgv --model=bgv-noise-mp24 --validate-noise --model=bgv-noise-mp24 input.mlir
```

### 4. Circuit Restructuring

- Reduce multiplication depth using `--operation-balancer`
- Consider approximation techniques for complex functions
- Split deep circuits into multiple phases with bootstrapping

## Integration with Parameter Generation

This pass works closely with parameter generation:

1. Parameter generation uses noise models to size parameters
1. Validation confirms the parameters are sufficient
1. If validation fails, parameters need to be regenerated with larger margins

## Performance Considerations

- Noise validation is typically fast (analysis only)
- Can be disabled in production builds if needed
- Debug output generation has minimal overhead
- IR annotation adds metadata but doesn't affect performance
