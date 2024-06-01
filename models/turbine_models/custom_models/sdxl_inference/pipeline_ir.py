tokens_to_image = r"""
module @sdxl_compiled_pipeline {{
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>) attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}]"}}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %arg1: tensor<{bd}x{max_length}x2048x{precision}>, %arg2: tensor<{bd}x1280x{precision}>, %arg3: tensor<{bd}x6x{precision}>, %arg4: tensor<{batch_size}x{precision}>, %arg5: tensor<{batch_size}xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}> attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]"}}
  func.func private @compiled_clip.encode_prompts(%arg0: tensor<{batch_size}x{max_length}xi64>, %arg1: tensor<{batch_size}x{max_length}xi64>, %arg2: tensor<{batch_size}x{max_length}xi64>, %arg3: tensor<{batch_size}x{max_length}xi64>) -> (tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>) attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}]"}}
  func.func private @{vae_fn_name}.main(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> tensor<{batch_size}x3x{width}x{height}x{precision}> attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]"}}

  func.func @tokens_to_image(%sample: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %guidance_scale: tensor<{batch_size}x{precision}>, %t_ids_1: tensor<{batch_size}x{max_length}xi64>, %t_ids_2: tensor<{batch_size}x{max_length}xi64>, %u_ids_1: tensor<{batch_size}x{max_length}xi64>, %u_ids_2: tensor<{batch_size}x{max_length}xi64>) -> tensor<{batch_size}x3x{width}x{height}x{precision}> {{
    %p_embeds, %t_embeds = func.call @compiled_clip.encode_prompts(%t_ids_1, %t_ids_2, %u_ids_1, %u_ids_2) : (tensor<{batch_size}x{max_length}xi64>, tensor<{batch_size}x{max_length}xi64>, tensor<{batch_size}x{max_length}xi64>, tensor<{batch_size}x{max_length}xi64>) -> (tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>)
    %noisy_sample, %time_ids, %steps = func.call @compiled_scheduled_unet.run_initialize(%sample) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps_int = tensor.extract %steps[] : tensor<i64>
    %n_steps = arith.index_cast %steps_int: i64 to index
    %res = scf.for %arg0 = %c0 to %n_steps step %c1 iter_args(%arg = %noisy_sample) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) {{
      %step_64 = arith.index_cast %arg0 : index to i64
      %this_step = tensor.from_elements %step_64 : tensor<1xi64>
      %inner = func.call @compiled_scheduled_unet.run_forward(%arg, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %this_step) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>, tensor<{bd}x6x{precision}>, tensor<{batch_size}x{precision}>, tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}>
      scf.yield %inner : tensor<{batch_size}x4x{lw}x{lh}x{precision}>
    }}
    %image = func.call @{vae_fn_name}.main(%res): (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> tensor<{batch_size}x3x{width}x{height}x{precision}>
    return %image : tensor<{batch_size}x3x{width}x{height}x{precision}>
  }}
}}
"""

unet_loop = r"""
module @sdxl_compiled_pipeline {{
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>) attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}]"}}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %arg1: tensor<{bd}x{max_length}x2048x{precision}>, %arg2: tensor<{bd}x1280x{precision}>, %arg3: tensor<{bd}x6x{precision}>, %arg4: tensor<{batch_size}x{precision}>, %arg5: tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}> attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]"}}
  
  func.func @produce_image_latents(%sample: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %p_embeds: tensor<{bd}x{max_length}x2048x{precision}>, %t_embeds: tensor<{bd}x1280x{precision}>, %guidance_scale: tensor<{batch_size}x{precision}>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}> {{
    %noisy_sample, %time_ids, %steps = func.call @compiled_scheduled_unet.run_initialize(%sample) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps_int = tensor.extract %steps[] : tensor<i64>
    %n_steps = arith.index_cast %steps_int: i64 to index
    %res = scf.for %arg0 = %c0 to %n_steps step %c1 iter_args(%arg = %noisy_sample) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) {{
      %step_64 = arith.index_cast %arg0 : index to i64
      %this_step = tensor.from_elements %step_64 : tensor<1xi64>
      %inner = func.call @compiled_scheduled_unet.run_forward(%arg, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %this_step) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>, tensor<{bd}x6x{precision}>, tensor<{batch_size}x{precision}>, tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}>
      scf.yield %inner : tensor<{batch_size}x4x{lw}x{lh}x{precision}>
    }}
    return %res : tensor<{batch_size}x4x{lw}x{lh}x{precision}>
  }}
}}
"""


def get_pipeline_ir(
    width: int,
    height: int,
    precision: str,
    batch_size: int,
    max_length: int,
    type: str,
    vae_fn_name: str = "compiled_vae",
):
    precision = "f32" if precision == "fp32" else "f16"
    if type == "tokens_to_image":
        return tokens_to_image.format(
            width=width,
            height=height,
            lw=int(width / 8),
            lh=int(height / 8),
            bd=int(batch_size * 2),
            precision=precision,
            batch_size=batch_size,
            max_length=max_length,
            vae_fn_name=vae_fn_name,
        )
    elif type == "unet_loop":
        return unet_loop.format(
            width=width,
            height=height,
            lw=int(width / 8),
            lh=int(height / 8),
            bd=int(batch_size * 2),
            precision=precision,
            batch_size=batch_size,
            max_length=max_length,
        )
