module {
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32> attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<1x4x128x128xf32>, %arg1: tensor<2x64x2048xf32>, %arg2: tensor<2x1280xf32>, %arg3: tensor<1xf32>, %arg4: i64) -> tensor<1x4x128x128xf32> attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"}
  
  func.func @produce_image_latents(%sample: tensor<1x4x128x128xf32>, %p_embeds: tensor<2x64x2048xf32>, %t_embeds: tensor<2x1280xf32>, %guidance_scale: tensor<1xf32>, %num_iters: index) -> tensor<1x4x128x128xf32> {
    %noisy_sample = func.call @compiled_scheduled_unet.run_initialize(%sample) : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %res = scf.for %arg0 = %c0 to %num_iters step %c1 iter_args(%arg = %noisy_sample) -> (tensor<1x4x128x128xf32>) {
      %step = arith.index_cast %arg0 : index to i64
      %inner = func.call @compiled_scheduled_unet.run_forward(%arg, %p_embeds, %t_embeds, %guidance_scale, %step) : (tensor<1x4x128x128xf32>, tensor<2x64x2048xf32>, tensor<2x1280xf32>, tensor<1xf32>, i64) -> tensor<1x4x128x128xf32>
      scf.yield %inner : tensor<1x4x128x128xf32>
    }
    return %res : tensor<1x4x128x128xf32>
  } 
}