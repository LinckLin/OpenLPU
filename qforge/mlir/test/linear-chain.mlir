// Linear-chain mechanism proof: qnn.matmul -> qisa.gemm (PF, M=128) and
// qnn.gemv (DC, M=1), qnn.rmsnorm -> qisa.rmsnorm.
// Run: mlir-opt --load-dialect-plugin=<path>/QnnPlugin.so --pass-pipeline='builtin.module(func.func(qnn-to-qisa))' \
//        qforge/mlir/test/linear-chain.mlir
module {

  // prefill linear chain: rmsnorm -> matmul -> matmul (M=128 -> GEMM)
  func.func @linear_chain_pf(%x: tensor<128x1024xf32>,
                             %w0: tensor<2048x1024xf32>,
                             %w1: tensor<1024x2048xf32>,
                             %gamma: tensor<2048xf32>) {
    %y0 = qnn.matmul %x, %w0 {group = 128 : i32}
          : tensor<128x1024xf32>, tensor<2048x1024xf32> -> tensor<128x2048xf32>
    %n = qnn.rmsnorm %y0, %gamma {eps = 1.000000e-06 : f32, mode = 0 : i32}
          : tensor<128x2048xf32>, tensor<2048xf32> -> tensor<128x2048xf32>
    %y1 = qnn.matmul %n, %w1 {group = 128 : i32}
          : tensor<128x2048xf32>, tensor<1024x2048xf32> -> tensor<128x1024xf32>
    return
  }

  // decode linear chain: matmul (M=1 -> GEMV)
  func.func @linear_chain_dc(%x: tensor<1x1024xf32>,
                             %w0: tensor<2048x1024xf32>) {
    %y0 = qnn.matmul %x, %w0 {group = 128 : i32}
          : tensor<1x1024xf32>, tensor<2048x1024xf32> -> tensor<1x2048xf32>
    return
  }

}
