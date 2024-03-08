module @sdxl_compiled_pipeline {
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<1x4x128x128xf16>) -> (tensor<1x4x128x128xf16>, tensor<2x6xf16>, tensor<i64>) attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]}]"}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<1x4x128x128xf16>, %arg1: tensor<2x64x2048xf16>, %arg2: tensor<2x1280xf16>, %arg3: tensor<2x6xf16>, %arg4: tensor<1xf16>, %arg5: tensor<1xi64>) -> tensor<1x4x128x128xf16> attributes {torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"}  
  func.func @produce_image_latents(%sample_0: tensor<1x4x128x128xf16>, %p_embeds: tensor<2x64x2048xf16>, %t_embeds: tensor<2x1280xf16>, %guidance_scale: tensor<1xf16>) -> tensor<1x4x128x128xf16> {
    %noisy_sample, %time_ids, %steps = func.call @compiled_scheduled_unet.run_initialize(%sample_0) : (tensor<1x4x128x128xf16>) -> (tensor<1x4x128x128xf16>, tensor<2x6xf16>, tensor<i64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %step_int = arith.index_cast %c0 : index to i64
    %step_inc_int = arith.index_cast %c1 : index to i64
    %step_0 = tensor.from_elements %step_int : tensor<1xi64>
    %step_inc = tensor.from_elements %step_inc_int : tensor<1xi64>
    %sample_1 = func.call @compiled_scheduled_unet.run_forward(%sample_0, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_0) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_1 = arith.addi %step_0, %step_inc : tensor<1xi64>
    %sample_2 = func.call @compiled_scheduled_unet.run_forward(%sample_1, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_1) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_2 = arith.addi %step_1, %step_inc : tensor<1xi64>
    %sample_3 = func.call @compiled_scheduled_unet.run_forward(%sample_2, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_2) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_3 = arith.addi %step_2, %step_inc : tensor<1xi64>
    %sample_4 = func.call @compiled_scheduled_unet.run_forward(%sample_3, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_3) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_4 = arith.addi %step_3, %step_inc : tensor<1xi64>
    %sample_5 = func.call @compiled_scheduled_unet.run_forward(%sample_4, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_4) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_5 = arith.addi %step_4, %step_inc : tensor<1xi64>
    %sample_6 = func.call @compiled_scheduled_unet.run_forward(%sample_5, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_5) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_6 = arith.addi %step_5, %step_inc : tensor<1xi64>
    %sample_7 = func.call @compiled_scheduled_unet.run_forward(%sample_6, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_6) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_7 = arith.addi %step_6, %step_inc : tensor<1xi64>
    %sample_8 = func.call @compiled_scheduled_unet.run_forward(%sample_7, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_7) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_8 = arith.addi %step_7, %step_inc : tensor<1xi64>
    %sample_9 = func.call @compiled_scheduled_unet.run_forward(%sample_8, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_8) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_9 = arith.addi %step_8, %step_inc : tensor<1xi64>
    %sample_10 = func.call @compiled_scheduled_unet.run_forward(%sample_9, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_9) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_10 = arith.addi %step_9, %step_inc : tensor<1xi64>
    %sample_11 = func.call @compiled_scheduled_unet.run_forward(%sample_10, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_10) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_11 = arith.addi %step_10, %step_inc : tensor<1xi64>
    %sample_12 = func.call @compiled_scheduled_unet.run_forward(%sample_11, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_11) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_12 = arith.addi %step_11, %step_inc : tensor<1xi64>
    %sample_13 = func.call @compiled_scheduled_unet.run_forward(%sample_12, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_12) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_13 = arith.addi %step_12, %step_inc : tensor<1xi64>
    %sample_14 = func.call @compiled_scheduled_unet.run_forward(%sample_13, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_13) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_14 = arith.addi %step_13, %step_inc : tensor<1xi64>
    %sample_15 = func.call @compiled_scheduled_unet.run_forward(%sample_14, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_14) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_15 = arith.addi %step_14, %step_inc : tensor<1xi64>
    %sample_16 = func.call @compiled_scheduled_unet.run_forward(%sample_15, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_15) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_16 = arith.addi %step_15, %step_inc : tensor<1xi64>
    %sample_17 = func.call @compiled_scheduled_unet.run_forward(%sample_16, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_16) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_17 = arith.addi %step_16, %step_inc : tensor<1xi64>
    %sample_18 = func.call @compiled_scheduled_unet.run_forward(%sample_17, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_17) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_18 = arith.addi %step_17, %step_inc : tensor<1xi64>
    %sample_19 = func.call @compiled_scheduled_unet.run_forward(%sample_18, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_18) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_19 = arith.addi %step_18, %step_inc : tensor<1xi64>
    %sample_20 = func.call @compiled_scheduled_unet.run_forward(%sample_19, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_19) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_20 = arith.addi %step_19, %step_inc : tensor<1xi64>
    %sample_21 = func.call @compiled_scheduled_unet.run_forward(%sample_20, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_20) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_21 = arith.addi %step_20, %step_inc : tensor<1xi64>
    %sample_22 = func.call @compiled_scheduled_unet.run_forward(%sample_21, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_21) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_22 = arith.addi %step_21, %step_inc : tensor<1xi64>
    %sample_23 = func.call @compiled_scheduled_unet.run_forward(%sample_22, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_22) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_23 = arith.addi %step_22, %step_inc : tensor<1xi64>
    %sample_24 = func.call @compiled_scheduled_unet.run_forward(%sample_23, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_23) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_24 = arith.addi %step_23, %step_inc : tensor<1xi64>
    %sample_25 = func.call @compiled_scheduled_unet.run_forward(%sample_24, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_24) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_25 = arith.addi %step_24, %step_inc : tensor<1xi64>
    %sample_26 = func.call @compiled_scheduled_unet.run_forward(%sample_25, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_25) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_26 = arith.addi %step_25, %step_inc : tensor<1xi64>
    %sample_27 = func.call @compiled_scheduled_unet.run_forward(%sample_26, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_26) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_27 = arith.addi %step_26, %step_inc : tensor<1xi64>
    %sample_28 = func.call @compiled_scheduled_unet.run_forward(%sample_27, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_27) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_28 = arith.addi %step_27, %step_inc : tensor<1xi64>
    %sample_29 = func.call @compiled_scheduled_unet.run_forward(%sample_28, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_28) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    %step_29 = arith.addi %step_28, %step_inc : tensor<1xi64>
    %sample_30 = func.call @compiled_scheduled_unet.run_forward(%sample_29, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %step_29) : (tensor<1x4x128x128xf16>, tensor<2x64x2048xf16>, tensor<2x1280xf16>, tensor<2x6xf16>, tensor<1xf16>, tensor<1xi64>) -> tensor<1x4x128x128xf16>
    return %sample_30 : tensor<1x4x128x128xf16>
  } 
}

