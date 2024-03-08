module @sdxl_compiled_pipeline {
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<1x4x128x128xf16>) -> (tensor<1x4x128x128xf16>, tensor<2x6xf16>, tensor<i64>) attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]}]"}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<2x64x2048xf16>, %arg2: tensor<2x1280xf16>, %arg3: tensor<2x6xf16>, %arg4: tensor<1xf16>, %arg5: tensor<1xi64>) -> tensor<1x4x128x128xf16> attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"}
  
  func.func @produce_image_latents(%sample_0: tensor<1x4x128x128xf16>, %p_embeds: tensor<2x64x2048xf16>, %t_embeds: tensor<2x1280xf16>, %guidance_scale: tensor<1xf16>) -> tensor<1x4x128x128xf16> {
    %noisy_sample, %time_ids, %steps = func.call @compiled_scheduled_unet.run_initialize(%sample_0) : (tensor<1x4x128x128xf16>) -> (tensor<1x4x128x128xf16>, tensor<2x6xf16>, tensor<i64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %step_int = arith.index_cast %c0 : index to i64
    %step_0 = tensor.from_elements %step_int : tensor<1xi64>
    %sample_1 = func.call @compiled_scheduled_unet.run_forward(%sample_0, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_0) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    return %sample_1 : tensor<1x4x128x128xf16>
  } 
}
