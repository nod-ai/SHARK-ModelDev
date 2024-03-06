module @sdxl_compiled_pipeline {
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16> attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<2x64x2048xf16>, %arg2: tensor<2x1280xf16>, %arg3: tensor<1xf16>, %arg4: tensor<1xi64>) -> tensor<1x4x128x128xf16> attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"}
  
  func.func @produce_image_latents(%sample: tensor<1x4x128x128xf16>, %p_embeds: tensor<2x64x2048xf16>, %t_embeds: tensor<2x1280xf16>, %guidance_scale: tensor<1xf16>, %steps_index: i32) -> tensor<1x4x128x128xf16> {
    %noisy_sample = func.call @compiled_scheduled_unet.run_initialize(%sample) : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n_steps = arith.index_cast %steps_index: i32 to index
    %res = scf.for %arg0 = %c0 to %n_steps step %c1 iter_args(%arg = %noisy_sample) -> (tensor<1x4x128x128xf16>) {
      %step_64 = arith.index_cast %arg0 : index to i64
      %this_step = tensor.from_elements %step_64 : tensor<1xi64>
      %inner = func.call @compiled_scheduled_unet.run_forward(%arg, %p_embeds, %t_embeds, %guidance_scale, %this_step) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
      scf.yield %inner : tensor<1x4x128x128xf16>
    }
    return %res : tensor<1x4x128x128xf16>
  } 
}