#ifndef LIB_TRANSFORMS_FORWARDINSERTTOEXTRACT_FORWARDINSERTTOEXTRACT_TD_
#define LIB_TRANSFORMS_FORWARDINSERTTOEXTRACT_FORWARDINSERTTOEXTRACT_TD_

include "mlir/Pass/PassBase.td"

def ForwardInsertToExtract : Pass<"forward-insert-to-extract"> {
    let summary = "Forward inserts to extracts within a single block";
    let description = [{
        This pass is similar to forward-store-to-load pass where store ops
        are forwarded load ops; here instead tensor.insert ops are forwarded
        to tensor.extract ops.

        Does not support complex control flow within a block, nor ops with
        arbitrary subregions.

        Developer Notes:
        - Tensor equivalent of forward-store-to-load for immutable tensor semantics
        - Essential for tensor optimization after loop unrolling in FHE pipelines
        - Eliminates redundant tensor.insert/tensor.extract pairs through direct value forwarding
        - Maintains tensor immutability while optimizing access patterns
        - Limitation: Single-block analysis only, no inter-block or complex control flow
        - Performance optimization: Reduces tensor creation overhead by direct value propagation
        - Common usage: After full-loop-unroll when tensor access patterns become statically clear
        - Reusable pattern: Core forwarding logic adaptable to other immutable data structure optimizations
        - Pipeline integration: Typically used with apply-folders for comprehensive tensor optimization

        Example transformation:
        ```mlir
        // Input: Insert followed by extract at same index
        %tensor1 = tensor.insert %value into %tensor0[%index] : tensor<16xi32>
        %extracted = tensor.extract %tensor1[%index] : tensor<16xi32>
        // ... use %extracted

        // Output: Direct value forwarding
        %tensor1 = tensor.insert %value into %tensor0[%index] : tensor<16xi32>
        // extract eliminated, replaced with direct use of %value
        // ... use %value
        ```
    }];
}

#endif // LIB_TRANSFORMS_FORWARDINSERTTOEXTRACT_FORWARDINSERTTOEXTRACT_TD_
