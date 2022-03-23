//===- AffineToStandard.h - Convert Affine to Standard dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineForOp;
class Location;
struct LogicalResult;
class OpBuilder;
class Pass;
class RewritePattern;
class Value;
class ValueRange;

class RewritePatternSet;



} // namespace mlir

namespace nngo {
/// Collect a set of patterns to convert from the Affine dialect to the Standard
/// dialect, in particular convert structured affine control flow into CFG
/// branch-based control flow.
void populateAffineToStdConversionPatterns(mlir::RewritePatternSet &patterns);

/// Collect a set of patterns to convert vector-related Affine ops to the Vector
/// dialect.
void populateAffineToVectorConversionPatterns(mlir::RewritePatternSet &patterns);

/// Emit code that computes the lower bound of the given affine loop using
/// standard arithmetic operations.
mlir::Value lowerAffineLowerBound(mlir::AffineForOp op, mlir::OpBuilder &builder);

/// Emit code that computes the upper bound of the given affine loop using
/// standard arithmetic operations.
mlir::Value lowerAffineUpperBound(mlir::AffineForOp op, mlir::OpBuilder &builder);

/// Lowers affine control flow operations (ForStmt, IfStmt and AffineApplyOp)
/// to equivalent lower-level constructs (flow of basic blocks and arithmetic
/// primitives).
std::unique_ptr<mlir::Pass> createLowerAffinePass();
}
