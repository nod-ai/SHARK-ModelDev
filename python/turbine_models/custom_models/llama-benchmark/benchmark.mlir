module {
  func.func private @state_update.run_initialize(%arg0: tensor<1x?xi64>) -> tensor<1x1xi64> attributes {
    torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"
  }
  func.func private @state_update.run_forward(%arg0: tensor<1x1xi64>) -> tensor<1x1xi64> attributes {
    torch.args_schema = "[1, {\22type\22: \22builtins.tuple\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: \22builtins.list\22, \22context\22: \22null\22, \22children_spec\22: [{\22type\22: null, \22context\22: null, \22children_spec\22: []}]}, {\22type\22: \22builtins.dict\22, \22context\22: \22[]\22, \22children_spec\22: []}]}]", torch.return_schema = "[1, {\22type\22: null, \22context\22: null, \22children_spec\22: []}]"
  }

  func.func @run(%input: tensor<1x?xi64>, %steps: tensor<i64>) -> tensor<1x1xi64> {
    %init = func.call @state_update.run_initialize(%input) : (tensor<1x?xi64>) -> tensor<1x1xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps_i64 = tensor.extract %steps[] : tensor<i64>
    %steps_index = arith.index_cast %steps_i64 : i64 to index
    %res = scf.for %arg0 = %c0 to %steps_index step %c1 iter_args(%arg = %init) -> (tensor<1x1xi64>) {
      %next = func.call @state_update.run_forward(%arg) : (tensor<1x1xi64>) -> tensor<1x1xi64>
      scf.yield %next : tensor<1x1xi64>
    }

    return %res : tensor<1x1xi64>
  }
}