tokens_to_image = r"""
module @sdxl_compiled_pipeline {{
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<{batch_size}x4x{lh}x{lw}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>) attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}]"}}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %arg1: tensor<{bd}x{max_length}x2048x{precision}>, %arg2: tensor<{bd}x1280x{precision}>, %arg3: tensor<{bd}x6x{precision}>, %arg4: tensor<1x{precision}>, %arg5: tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}> attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]"}}
  func.func private @compiled_clip.encode_prompts(%arg0: tensor<{batch_size}x{max_length}xi64>, %arg1: tensor<{batch_size}x{max_length}xi64>, %arg2: tensor<{batch_size}x{max_length}xi64>, %arg3: tensor<{batch_size}x{max_length}xi64>) -> (tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>) attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}]"}}
  func.func private @{vae_module}.main(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> tensor<{batch_size}x3x{width}x{height}x{precision}> attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]"}}

  func.func @tokens_to_image(%sample: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %guidance_scale: tensor<1x{precision}>, %t_ids_1: tensor<{batch_size}x{max_length}xi64>, %t_ids_2: tensor<{batch_size}x{max_length}xi64>, %u_ids_1: tensor<{batch_size}x{max_length}xi64>, %u_ids_2: tensor<{batch_size}x{max_length}xi64>) -> tensor<{batch_size}x3x{width}x{height}x{precision}> {{
    %p_embeds, %t_embeds = func.call @compiled_clip.encode_prompts(%t_ids_1, %t_ids_2, %u_ids_1, %u_ids_2) : (tensor<{batch_size}x{max_length}xi64>, tensor<{batch_size}x{max_length}xi64>, tensor<{batch_size}x{max_length}xi64>, tensor<{batch_size}x{max_length}xi64>) -> (tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>)
    %noisy_sample, %time_ids, %steps = func.call @compiled_scheduled_unet.run_initialize(%sample) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps_int = tensor.extract %steps[] : tensor<i64>
    %n_steps = arith.index_cast %steps_int: i64 to index
    %res = scf.for %arg0 = %c0 to %n_steps step %c1 iter_args(%arg = %noisy_sample) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) {{
      %step_64 = arith.index_cast %arg0 : index to i64
      %this_step = tensor.from_elements %step_64 : tensor<1xi64>
      %inner = func.call @compiled_scheduled_unet.run_forward(%arg, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %this_step) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>, tensor<{bd}x6x{precision}>, tensor<1x{precision}>, tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}>
      scf.yield %inner : tensor<{batch_size}x4x{lw}x{lh}x{precision}>
    }}
    %image = func.call @{vae_module}.main(%res): (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> tensor<{batch_size}x3x{width}x{height}x{precision}>
    return %image : tensor<{batch_size}x3x{width}x{height}x{precision}>
  }}
}}
"""

unet_loop = r"""
module @sdxl_compiled_pipeline {{
  func.func private @compiled_scheduled_unet.run_initialize(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>) attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}]"}}
  func.func private @compiled_scheduled_unet.run_forward(%arg0: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %arg1: tensor<{bd}x{max_length}x2048x{precision}>, %arg2: tensor<{bd}x1280x{precision}>, %arg3: tensor<{bd}x6x{precision}>, %arg4: tensor<1x{precision}>, %arg5: tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}> attributes {{torch.args_schema = "[1, {{\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]}}, {{\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}}]}}]", torch.return_schema = "[1, {{\22type\22: null, \22context\22: null, \22children_spec\22: []}}]"}}
  
  func.func @produce_image_latents(%sample: tensor<{batch_size}x4x{lw}x{lh}x{precision}>, %p_embeds: tensor<{bd}x{max_length}x2048x{precision}>, %t_embeds: tensor<{bd}x1280x{precision}>, %guidance_scale: tensor<1x{precision}>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}> {{
    %noisy_sample, %time_ids, %steps = func.call @compiled_scheduled_unet.run_initialize(%sample) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x6x{precision}>, tensor<i64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps_int = tensor.extract %steps[] : tensor<i64>
    %n_steps = arith.index_cast %steps_int: i64 to index
    %res = scf.for %arg0 = %c0 to %n_steps step %c1 iter_args(%arg = %noisy_sample) -> (tensor<{batch_size}x4x{lw}x{lh}x{precision}>) {{
      %step_64 = arith.index_cast %arg0 : index to i64
      %this_step = tensor.from_elements %step_64 : tensor<1xi64>
      %inner = func.call @compiled_scheduled_unet.run_forward(%arg, %p_embeds, %t_embeds, %time_ids, %guidance_scale, %this_step) : (tensor<{batch_size}x4x{lw}x{lh}x{precision}>, tensor<{bd}x{max_length}x2048x{precision}>, tensor<{bd}x1280x{precision}>, tensor<{bd}x6x{precision}>, tensor<1x{precision}>, tensor<1xi64>) -> tensor<{batch_size}x4x{lw}x{lh}x{precision}>
      scf.yield %inner : tensor<{batch_size}x4x{lw}x{lh}x{precision}>
    }}
    return %res : tensor<{batch_size}x4x{lw}x{lh}x{precision}>
  }}
}}
"""

produce_img_split = r"""
module @sdxl_compiled_pipeline {{
  func.func private @{scheduler_module}.run_initialize(%arg0: !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) -> (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, !torch.vtensor<[{bd},6],{precision}>, !torch.vtensor<[1],f16>, !torch.vtensor<[{num_steps}],f32>) attributes {{torch.assume_strict_symbolic_shapes}}
  func.func private @{scheduler_module}.run_scale(%arg0: !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, %arg1: !torch.vtensor<[1],si64>, %arg2: !torch.vtensor<[{num_steps}],f32>) -> (!torch.vtensor<[{bd},4,{lh},{lw}],{precision}>, !torch.vtensor<[1],{precision}>) attributes {{torch.assume_strict_symbolic_shapes}}
  func.func private @{scheduler_module}.run_step(%arg0: !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, %arg1: !torch.vtensor<[1],{precision}>, %arg2: !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) -> !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}> attributes {{torch.assume_strict_symbolic_shapes}}
  func.func private @{unet_module}.{unet_function}(%arg0: !torch.vtensor<[{bd},4,{lh},{lw}],{precision}>, %arg1: !torch.vtensor<[1],{precision}>, %arg2: !torch.vtensor<[{bd},{max_length},2048],{precision}>, %arg3: !torch.vtensor<[{bd},1280],{precision}>, %arg4: !torch.vtensor<[{bd},6],{precision}>, %arg5: !torch.vtensor<[1],{precision}>) -> !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}> attributes {{torch.assume_strict_symbolic_shapes}}
  func.func private @{vae_module}.decode(%arg0: !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) -> !torch.vtensor<[{batch_size},3,{height},{width}],{precision}> attributes {{torch.assume_strict_symbolic_shapes}}
  
  func.func @produce_image_latents(%sample: !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, %p_embeds: !torch.vtensor<[{bd},{max_length},2048],{precision}>, %t_embeds: !torch.vtensor<[{bd},1280],{precision}>, %guidance_scale: !torch.vtensor<[1],{precision}>) -> !torch.vtensor<[{batch_size},3,{height},{width}],{precision}> {{
    %noisy_sample, %time_ids, %delete, %timesteps = func.call @{scheduler_module}.run_initialize(%sample) : (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) -> (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, !torch.vtensor<[{bd},6],{precision}>, !torch.vtensor<[1],{precision}>, !torch.vtensor<[{num_steps}],f32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n_steps = arith.constant {num_steps} : index
    %res = scf.for %arg0 = %c0 to %n_steps step %c1 iter_args(%arg = %noisy_sample) -> (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) {{
      %step_64 = arith.index_cast %arg0 : index to i64
      %this_step = tensor.from_elements %step_64 : tensor<1xi64>
      %step_torch = torch_c.from_builtin_tensor %this_step : tensor<1xi64> -> !torch.vtensor<[1],si64>
      %scaled, %timestep = func.call @{scheduler_module}.run_scale(%arg, %step_torch, %timesteps) : (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, !torch.vtensor<[1],si64>, !torch.vtensor<[{num_steps}],f32>) -> (!torch.vtensor<[{bd},4,{lh},{lw}],{precision}>, !torch.vtensor<[1],{precision}>)
      %inner = func.call @{unet_module}.{unet_function}(%scaled, %timestep, %p_embeds, %t_embeds, %time_ids, %guidance_scale) : (!torch.vtensor<[{bd},4,{lh},{lw}],{precision}>, !torch.vtensor<[1],{precision}>, !torch.vtensor<[{bd},{max_length},2048],{precision}>, !torch.vtensor<[{bd},1280],{precision}>, !torch.vtensor<[{bd},6],{precision}>, !torch.vtensor<[1],{precision}>) -> !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>
      %pred = func.call @{scheduler_module}.run_step(%inner, %timestep, %arg) : (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>, !torch.vtensor<[1],{precision}>, !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) -> !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>
      scf.yield %pred : !torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>
    }}
    %image = func.call @{vae_module}.decode(%res): (!torch.vtensor<[{batch_size},4,{lh},{lw}],{precision}>) -> !torch.vtensor<[{batch_size},3,{height},{width}],{precision}>
    return %image : !torch.vtensor<[{batch_size},3,{height},{width}],{precision}>
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
    num_steps: int = 20,
    vae_module: str = "compiled_vae",
    unet_module_name: str = "compiled_punet",
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
            vae_module=vae_module,
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
    elif type == "produce_img_split":
        unet_fn_name = "run_forward"
        scheduler_module_name = "compiled_scheduler"
        vae_module_name = "compiled_vae"
        return produce_img_split.format(
            width=width,
            height=height,
            lw=int(width / 8),
            lh=int(height / 8),
            bd=int(batch_size * 2),
            precision=precision,
            batch_size=batch_size,
            max_length=max_length,
            unet_module=unet_module_name,
            unet_function=unet_fn_name,
            scheduler_module=scheduler_module_name,
            vae_module=vae_module_name,
            num_steps=num_steps,
        )