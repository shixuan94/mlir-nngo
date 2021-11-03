// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

//func private @test_transpose(%arg0 : tensor<10x1xf32>) -> tensor<*xf32> {
//  %0 = "onnx.Transpose"(%arg0) : (tensor<10x1xf32>) -> tensor<*xf32>
//  "std.return"(%0) : (tensor<*xf32>) -> ()
//}

// Test whether the lowering is correct in the presence of static dimensions.
func private @test_category_mapper(%arg0 : tensor<2x!onnx.String>) -> tensor<2xi64> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_int64 = 0: si64} : (tensor<2x!onnx.String>) -> tensor<2xi64>
  "std.return"(%0) : (tensor<2xi64>) -> ()
}

// Test whether the lowering is correct in the presence of dynamic dimensions.
//func private @test_depth_to_space_dynamic_dims1(%arg0 : tensor<1x?x8x?xi64>) -> tensor<1x?x8x?x!onnx.String> {
//  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["rain", "cat", "dog"], default_string = "???"} : (tensor<1x?x8x?xi64>) -> tensor<1x?x8x?x!onnx.String>
//  "std.return"(%0) : (tensor<1x?x8x?x!onnx.String>) -> ()
//}