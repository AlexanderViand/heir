---
title: Propagate Annotation
weight: 34
---

## Overview

The propagate-annotation pass propagates attributes from one operation to
subsequent (or preceding) operations throughout the IR. This is useful for
maintaining metadata, analysis results, or optimization hints as they flow
through the computation graph. The pass ensures that important attributes are
preserved and available where needed.

## Input/Output

- **Input**: IR with operations containing attributes that need to be propagated
- **Output**: IR where specified attributes are propagated to operations that
  don't already have them

## Options

- `--attr-name=<string>`: The attribute name to propagate (required)
- `--reverse=<bool>`: Whether to propagate in reverse direction (default: false)

## Usage Examples

```bash
heir-opt --propagate-annotation=attr-name=test.attr input.mlir
```

Propagate in reverse direction:

```bash
heir-opt --propagate-annotation=attr-name=test.attr,reverse=true input.mlir
```

## When to Use

Use this pass when you need to:

1. Maintain analysis results throughout the IR
1. Propagate optimization hints or metadata
1. Ensure attributes are available for downstream passes
1. Maintain debugging or profiling information
1. Propagate security or privacy annotations

Typical placement in compilation pipelines:

1. After analysis passes that generate important attributes
1. Before passes that require specific attributes to be present
1. Throughout the pipeline to maintain important metadata
1. After transformations that might lose attribute information

## How It Works

The pass operates through dataflow propagation:

1. **Attribute Detection**: Identifies operations with the specified attribute
1. **Direction Selection**: Propagates forward or backward based on the reverse
   option
1. **Propagation Logic**: Copies attributes to operations that don't already
   have them
1. **Preservation**: Avoids overwriting existing attributes with the same name

## Example

**Forward propagation - Before:**

```mlir
func.func @foo(%arg0: i16 {test.attr = 1}) -> i16 {
  %0 = arith.muli %arg0, %arg0 : i16
  %1 = mgmt.relinearize %0 : i16
  return %1 : i16
}
```

**Forward propagation - After:**

```mlir
func.func @foo(%arg0: i16 {test.attr = 1 : i64}) -> i16 {
  %0 = arith.muli %arg0, %arg0 {test.attr = 1 : i64} : i16
  %1 = mgmt.relinearize %0 {test.attr = 1 : i64} : i16
  return {test.attr = 1 : i64} %1 : i16
}
```

## Use Cases

### Security Annotations

Propagate security levels or encryption parameters:

```bash
heir-opt --propagate-annotation=attr-name=security.level input.mlir
```

### Analysis Results

Maintain analysis results like noise levels or multiplicative depth:

```bash
heir-opt --propagate-annotation=attr-name=noise.level input.mlir
```

### Optimization Hints

Propagate optimization hints for later passes:

```bash
heir-opt --propagate-annotation=attr-name=opt.hint input.mlir
```

## Benefits

- **Metadata Preservation**: Maintains important attributes throughout
  transformations
- **Analysis Support**: Enables passes to rely on propagated analysis results
- **Debugging Aid**: Helps track information flow for debugging
- **Flexible Direction**: Supports both forward and backward propagation

## Propagation Rules

1. **Non-Overwriting**: Does not overwrite existing attributes with the same
   name
1. **Selective**: Only propagates the specified attribute name
1. **Conditional**: Only adds attributes to operations that don't already have
   them
1. **Directional**: Respects the forward/reverse propagation setting

## Advanced Usage

### Multiple Attributes

Run multiple times for different attributes:

```bash
heir-opt --propagate-annotation=attr-name=attr1 \
         --propagate-annotation=attr-name=attr2 \
         input.mlir
```

### Bidirectional Propagation

Combine forward and reverse passes:

```bash
heir-opt --propagate-annotation=attr-name=test.attr \
         --propagate-annotation=attr-name=test.attr,reverse=true \
         input.mlir
```

## Limitations

- Only propagates one attribute type per pass invocation
- Does not perform semantic analysis of attribute compatibility
- May propagate attributes to operations where they're not meaningful
- Respects existing attributes (no overwriting)

## Related Passes

- Often used after analysis passes that generate attributes
- Works well with attribute-aware optimization passes
- Can be combined with attribute validation passes
- May be used before passes that require specific attribute annotations
