add_onnx_mlir_library(OMAffineToStandard
  AffineToStandard.cpp

  # DEPENDS
  # MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineUtils
  MLIRArithmetic
  MLIRIR
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRStandard
  MLIRTransforms
  MLIRVector
  )