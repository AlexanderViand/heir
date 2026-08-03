#include "lib/Dialect/Cheddar/Transforms/FreeIntermediates.h"

#include <iterator>
#include <utility>

#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarTypes.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"              // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CHEDDARFREEINTERMEDIATES
#include "lib/Dialect/Cheddar/Transforms/FreeIntermediates.h.inc"

namespace {

// Is this a memref of a move-only CHEDDAR payload (owns GPU memory)?
static bool isPayloadMemRef(Type t) {
  if (!isa<MemRefType>(t)) return false;
  return isa<CiphertextType, PlaintextType, LinearTransformType>(
      cast<MemRefType>(t).getElementType());
}

// Does `op` declare only Read effects on `v`? (Ops without the effect
// interface are treated as unknown -> false.)
static bool onlyReads(Operation* op, Value v) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface) return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffectsOnValue(v, effects);
  return llvm::all_of(effects, [](const MemoryEffects::EffectInstance& e) {
    return isa<MemoryEffects::Read>(e.getEffect());
  });
}

// Is `buf` never written and never aliased in its enclosing function? Only
// direct loads and declared-read-only uses qualify; any store, view, cast, or
// unknown user disqualifies it.
static bool bufferIsReadOnly(Value buf) {
  return llvm::all_of(buf.getUsers(), [&](Operation* user) {
    if (auto load = dyn_cast<memref::LoadOp>(user))
      return load.getMemref() == buf;
    return onlyReads(user, buf);
  });
}

// Bufferization stages a payload read out of a read-only buffer (typically the
// caller-owned split-preprocessing storage argument) through a local alloc:
//
//   %v = memref.load %src[%i]      // %src never written in this function
//   %alloc = memref.alloc() : memref<!pt>
//   memref.store %v, %alloc[]
//   cheddar.op ..., %alloc, ...    // read-only users
//
// Payloads are move-only in C++, so CheddarToEmitC must render the staging
// store as `local = std::move(src[i])` -- destroying the CALLER's element when
// %src is a function argument: a second call over the same storage reads
// moved-from payloads ("Plaintext num primes mismatch" at runtime). Forward
// the readers to a rank-reducing subview of %src[%i] and drop the staging
// alloc entirely; %src stays intact (and, now unwritten, becomes a const
// argument in the emitted C++).
static void forwardReadOnlyStagedPayloads(Block* block) {
  // Positions for store-dominates-readers checks without O(n^2)
  // isBeforeInBlock.
  DenseMap<Operation*, unsigned> pos;
  unsigned n = 0;
  for (Operation& op : *block) pos[&op] = n++;

  SmallVector<memref::AllocOp> candidates;
  for (Operation& op : *block)
    if (auto alloc = dyn_cast<memref::AllocOp>(&op))
      if (isPayloadMemRef(alloc.getType()) && alloc.getType().getRank() == 0)
        candidates.push_back(alloc);

  for (memref::AllocOp alloc : candidates) {
    Value buf = alloc.getMemref();
    memref::StoreOp store;
    SmallVector<memref::LoadOp> loads;
    SmallVector<Operation*> readers;
    bool ok = true;
    for (Operation* user : buf.getUsers()) {
      // ponytail: same-block, straight-line only (post-unroll/inline code is
      // flat); nested-region readers just keep the staging copy.
      if (user->getBlock() != block) {
        ok = false;
        break;
      }
      if (auto s = dyn_cast<memref::StoreOp>(user)) {
        if (s.getMemref() != buf || store) {  // value==buf, or a second store
          ok = false;
          break;
        }
        store = s;
      } else if (auto l = dyn_cast<memref::LoadOp>(user)) {
        // A loaded payload value that feeds a store/copy elsewhere would just
        // relocate the move; only pure readers of the value qualify.
        for (Operation* vu : l.getResult().getUsers())
          if (isa<memref::StoreOp, memref::CopyOp>(vu)) ok = false;
        if (!ok) break;
        loads.push_back(l);
      } else if (onlyReads(user, buf)) {
        readers.push_back(user);
      } else {
        ok = false;
        break;
      }
    }
    if (!ok || !store) continue;

    auto srcLoad = store.getValueToStore().getDefiningOp<memref::LoadOp>();
    if (!srcLoad || srcLoad->getBlock() != block) continue;
    Value src = srcLoad.getMemref();
    if (src == buf || !bufferIsReadOnly(src)) continue;

    // The store must dominate every reader.
    unsigned storePos = pos.lookup(store);
    auto after = [&](Operation* r) { return pos.lookup(r) > storePos; };
    if (!llvm::all_of(readers, after) ||
        !llvm::all_of(loads, [&](memref::LoadOp l) { return after(l); }))
      continue;

    // Readers taking the buffer as an operand get a rank-reducing subview of
    // the source slot, strided type and all. No cast back to the alloc's
    // identity-layout type: the readers here are cheddar DPS ops (the only
    // ops that pass the effect filter), whose operands accept any layout --
    // and a cast would be composed with the canonicalizer's own
    // subview-constant-folding cast by foldMemRefCast (which skips the
    // compatibility check) into an invalid static-offset-to-identity cast.
    if (!readers.empty()) {
      OpBuilder b(store);
      Location loc = alloc.getLoc();
      auto srcTy = cast<MemRefType>(src.getType());
      SmallVector<OpFoldResult> offsets(llvm::map_range(
          srcLoad.getIndices(), [](Value v) { return OpFoldResult(v); }));
      SmallVector<OpFoldResult> ones(srcTy.getRank(), b.getIndexAttr(1));
      MemRefType svTy = memref::SubViewOp::inferRankReducedResultType(
          /*resultShape=*/{}, srcTy, offsets, ones, ones);
      Value slot =
          memref::SubViewOp::create(b, loc, svTy, src, offsets, ones, ones);
      for (Operation* r : readers) r->replaceUsesOfWith(buf, slot);
    }
    // Readers of the loaded value reuse the source load's result directly.
    for (memref::LoadOp l : loads) {
      l.getResult().replaceAllUsesWith(srcLoad.getResult());
      l.erase();
    }
    store.erase();
    alloc.erase();
    if (srcLoad->use_empty()) srcLoad.erase();
  }
}

struct CheddarFreeIntermediates
    : impl::CheddarFreeIntermediatesBase<CheddarFreeIntermediates> {
  void runOnOperation() override {
    // (buffer, op-after-which-to-free). Collected during analysis; deallocs are
    // inserted only afterwards so we never invalidate block ordering /
    // iterators mid-analysis (which would make this quadratic).
    SmallVector<std::pair<Value, Operation*>> toFree;

    // Process each block independently with a single linear forward sweep, so
    // last-use is found by op position without any O(n) isBeforeInBlock calls.
    getOperation()->walk([&](Block* block) {
      // First forward staging copies of read-only buffers so a caller-owned
      // payload is never moved from (and the staging alloc is never freed --
      // it no longer exists).
      forwardReadOnlyStagedPayloads(block);

      // Payload buffers allocated in *this* block (the only freeable locals).
      DenseSet<Value> localAllocs;
      bool alreadyFreed = false;
      for (Operation& op : *block) {
        if (auto a = dyn_cast<memref::AllocOp>(&op)) {
          Value buf = a.getMemref();
          if (isPayloadMemRef(buf.getType())) localAllocs.insert(buf);
        }
        if (isa<memref::DeallocOp>(op)) alreadyFreed = true;
      }
      if (localAllocs.empty()) return;

      // Last top-level op in this block that (transitively) uses each local
      // alloc. Walking each top-level op's subtree once attributes nested uses
      // (e.g. inside an scf.for body) to the enclosing block-level op.
      DenseMap<Value, Operation*> lastUse;
      for (Operation& top : *block) {
        top.walk([&](Operation* inner) {
          for (Value operand : inner->getOperands())
            if (localAllocs.contains(operand)) lastUse[operand] = &top;

          // Split-preprocessing stores an encoded payload into persistent
          // storage as bufferized `%v = memref.load %localAlloc[] ;
          // memref.store %v, %storage[...]` (the rank-0 tensor.extract -> store
          // from PreprocessingToCheddar). The store's operands are %v and
          // %storage, NOT %localAlloc, so the operand sweep above would treat
          // the memref.load as the alloc's last use and free it BEFORE the
          // store -- leaving an EMPTY payload (NPInfo 0,0,0) in storage. Treat
          // the store into a non-local payload memref as a use of the loaded
          // local alloc so its dealloc lands after the store.
          if (auto store = dyn_cast<memref::StoreOp>(inner)) {
            Value dst = store.getMemref();
            if (!localAllocs.contains(dst) && isPayloadMemRef(dst.getType())) {
              if (auto load =
                      store.getValueToStore().getDefiningOp<memref::LoadOp>()) {
                Value src = load.getMemref();
                if (localAllocs.contains(src)) lastUse[src] = &top;
              }
            }
          }
        });
      }

      for (Value buf : localAllocs) {
        auto it = lastUse.find(buf);
        if (it == lastUse.end()) continue;  // unused -> leave to scope cleanup
        toFree.push_back({buf, it->second});
      }
      // If any explicit dealloc already exists in the block, skip those buffers
      // (a prior dealloc pass ran); cheap guard since it's rare here.
      if (alreadyFreed) {
        for (auto& pair : toFree)
          for (Operation* u : pair.first.getUsers())
            if (isa<memref::DeallocOp>(u)) {
              pair.second = nullptr;
              break;
            }
      }
    });

    for (auto& [buf, after] : toFree) {
      if (!after) continue;
      OpBuilder b(after->getBlock(), std::next(after->getIterator()));
      memref::DeallocOp::create(b, buf.getLoc(), buf);
    }
  }
};

}  // namespace
}  // namespace mlir::heir::cheddar
