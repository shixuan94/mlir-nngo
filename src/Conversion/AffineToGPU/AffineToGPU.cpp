/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- LowerKrnl.cpp - Krnl Dialect Lowering -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"
#include "src/Support/KrnlSupport.hpp"

#include "llvm/Support/Debug.h"

#include <functional>
#include <mutex>

static constexpr int BUFFER_ALIGN = 64;

#define DEBUG_TYPE "affine_to_gpu"

using namespace mlir;

namespace {

struct ConvertAffineToGPUPass
    : public PassWrapper<ConvertAffineToGPUPass, OperationPass<FuncOp>> {

  StringRef getArgument() const override { return "convert-affine-to-gpu"; }

  StringRef getDescription() const override { return "Lower affine dialect."; }

  void runOnOperation() final;
};


void ConvertKrnlToAffinePass::runOnOperation() {
  OpBuilder builder(&getContext());
  FuncOp funcOp = getOperation();

  // external function: nothing to do
  if (funcOp.body().empty()) {
    return;
  }


  ConversionTarget target(getContext());
  // Legal/illegal ops.
  target.addIllegalOp<KrnlTerminatorOp>();
  // krnl.dim operations must be lowered prior to this pass.
  target.addIllegalOp<KrnlDimOp>();
  target.addIllegalOp<KrnlMatMulOp>();
  target.addIllegalOp<KrnlCopyToBufferOp>();
  target.addIllegalOp<KrnlCopyFromBufferOp>();
  target.addIllegalOp<KrnlMemsetOp>();
  target.addLegalOp<AffineYieldOp>();
  target.addLegalOp<AffineLoadOp>();
  target.addLegalOp<AffineStoreOp>();
  target.addLegalOp<KrnlVectorTypeCastOp>();
  target.addLegalDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
      mlir::memref::MemRefDialect, mlir::StandardOpsDialect,
      mlir::vector::VectorDialect>();
  // Patterns.
  RewritePatternSet patterns(&getContext());
  patterns.insert<KrnlTerminatorLowering>(&getContext());
  patterns.insert<KrnlLoadLowering>(&getContext());
  patterns.insert<KrnlStoreLowering>(&getContext());
  patterns.insert<KrnlMatmulLowering>(&getContext());
  patterns.insert<KrnlCopyToBufferLowering>(&getContext());
  patterns.insert<KrnlCopyFromBufferLowering>(&getContext());
  patterns.insert<KrnlMemsetLowering>(&getContext());

  if (failed(applyPartialConversion(
          getOperation(), target, std::move(patterns), &unconverted))) {
    signalPassFailure();
    return;
  }

}

} // namespace

std::unique_ptr<Pass> onnx_mlir::createConvertAffineToGPUPass() {
  return std::make_unique<ConvertAffineToGPUPass>();
}