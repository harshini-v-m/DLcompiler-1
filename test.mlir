//test file
func.func @test_add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
 %0 = nova.add %arg0, %arg1 : tensor<1xf32>, tensor<1xf32> 
 %1 = nova.sub %arg0, %arg1 : tensor<1xf32>, tensor<1xf32>
 %3=nova.neg  %0 : tensor<1xf32>
  return %0 : tensor<1xf32>
}

