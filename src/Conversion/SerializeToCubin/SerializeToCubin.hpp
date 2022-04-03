//===- TestConvertGPUKernelToCubin.cpp - Test gpu kernel cubin lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace {
class TestSerializeToCubinPass
    : public PassWrapper<TestSerializeToCubinPass, gpu::SerializeToBlobPass> {
public:
  StringRef getArgument() const final { return "test-gpu-to-cubin"; }
  StringRef getDescription() const final {
    return "Lower GPU kernel function to CUBIN binary annotations";
  }
  TestSerializeToCubinPass();

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Serializes PTX to CUBIN.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;
};
} // namespace

TestSerializeToCubinPass::TestSerializeToCubinPass() {
  this->triple = "nvptx64-nvidia-cuda";
  this->chip = "sm_35";
  this->features = "+ptx60";
}

void TestSerializeToCubinPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerNVVMDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<std::vector<char>>
TestSerializeToCubinPass::serializeISA(const std::string &) {
  std::string data = "CUBIN";
  return std::make_unique<std::vector<char>>(data.begin(), data.end());
}

namespace mlir {

std::unique_ptr<Pass> createSerializeToCubin() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  return std::make_unique<TestSerializeToCubinPass>();
}
} // namespace mlir
