# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMAffineToGPU
  AffineToGPU.cpp

  LINK_LIBS PUBLIC
  OMSpecializedKernelOpInterface
  OMONNXOps
  OMSupport
  MLIRTransforms
  MLIRAffineUtils
  )
