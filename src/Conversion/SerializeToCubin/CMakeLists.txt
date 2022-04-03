# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSerializeToCubin
  SerializeToCubin.hpp

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRGPUOps
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMIR
  MLIRNVVMToLLVMIRTranslation
  MLIRPass
  MLIRROCDLIR
  MLIRROCDLToLLVMIRTranslation
  MLIRSCF
  MLIRStandard
  MLIRTransformUtils
  )
