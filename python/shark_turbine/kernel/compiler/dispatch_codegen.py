"""Code generation support for top-level IREE dispatch constructs.

This assumes that you have some form of code generation for the
"inside" of some kernels, as this layer is responsible for
embedding and generating the calls/dispatches.
"""

from typing import Callable, Optional

from .builder import (
    ModuleBuilder,
)

from .ir import (
    Block,
    FunctionType,
    IndexType,
    InsertionPoint,
    IrType,
    Location,
    Operation,
    StringAttr,
    Value,
    func_d,
    stream_d,
)


class StreamExecutable:
    """Encapsulates a 'stream' compilable executable which can be dispatched to.

    This corresponds to a `stream.executable`, consisting of one or more exported
    dispatch functions.
    """

    __slots__ = [
        "_mb",
        "_exe_op",
        "_exe_block",
        "_loc",
        "sym_name",
        "def_module",
    ]

    def __init__(
        self,
        mb: ModuleBuilder,
        *,
        loc: Optional[Location] = None,
        name: str = "__executable"
    ):
        self._mb = mb
        if not loc:
            loc = mb.unknown_loc
        self._loc = loc

        # Construct the executable.
        with loc:
            with InsertionPoint(mb.body_block):
                self._exe_op = exe_op = stream_d.ExecutableOp(
                    name, sym_visibility="private"
                )
                exe_block = exe_op.body.blocks.append()
                self._exe_block: Block = exe_block
                stream_d.ExecutableEndOp(ip=InsertionPoint(exe_block))
            mb.symbol_table.insert(exe_op)
            self.sym_name: StringAttr = exe_op.sym_name

            # Construct the inner definitions module.
            with InsertionPoint.at_block_begin(exe_block):
                self.def_module = ModuleBuilder(context=mb.context)

    def define_entrypoint(
        self,
        name: str,
        arg_arity: int,
        workload_arity: int,
        result_arity: int,
        grid_arity: int = 3,
    ) -> tuple["WorkgroupCalcBuilder", "DispatchFuncBuilder"]:
        """Defines a dispatch function with a signature like:

        ```
        func.func @name(%in0 : !stream.binding, %in1 : !stream.binding,
                        %workload0 : index, %workload1 : index,
                        %result0 : !stream.binding, %result1 : !stream.binding)
        ```

        Also adds an export with workgroup function like:

        ```
        stream.executable.export public @name(%workload0 : index, %workload1 : index) -> (index, [[grid_arity...]]) {

        }
        ```

        The given name is not uniqued (must be unique as given by the caller).
        """
        with self._loc:
            binding_type = IrType.parse("!stream.binding")
            index_type = IndexType.get()

            # Define the dispatch function.
            def_ftype = FunctionType.get(
                (
                    (arg_arity * [binding_type])
                    + (workload_arity * [index_type])
                    + (result_arity * [binding_type])
                ),
                [],
            )
            with InsertionPoint(self.def_module.body_block):
                def_func_op = func_d.FuncOp(name, def_ftype)
                def_func_block = def_func_op.add_entry_block()
                def_func_args = list(def_func_block.arguments)

            # Define the export.
            with InsertionPoint.at_block_begin(self._exe_block):
                export_op = stream_d.ExecutableExportOp(name, name)
                export_block = export_op.workgroup_count.blocks.append(
                    *(workload_arity * [index_type])
                )

        return (
            WorkgroupCalcBuilder(export_block, lambda vs: stream_d.ReturnOp(vs)),
            DispatchFuncBuilder(
                def_func_block,
                def_func_args[0:arg_arity],
                def_func_args[arg_arity : (arg_arity + workload_arity)],
                def_func_args[(arg_arity + workload_arity) :],
            ),
        )


class WorkgroupCalcBuilder:
    """Builder for a workgroup calculation block."""

    __slots__ = [
        "entry_block",
        "workload",
        "_term_ctor",
    ]

    def __init__(self, entry_block: Block, term_ctor: Callable[[list[Value]], None]):
        self.entry_block = entry_block
        self.workload = list(entry_block.arguments)
        self._term_ctor = term_ctor

    @property
    def location(self) -> Location:
        return self.entry_block.owner.location

    def terminate(self, returns: list[Value]):
        entry_block = self.entry_block
        with entry_block.owner.location, InsertionPoint(entry_block):
            self._term_ctor(returns)


class DispatchFuncBuilder:
    """Builder for dispatch functions."""

    __slots__ = [
        "entry_block",
        "args",
        "workload",
        "results",
    ]

    def __init__(
        self,
        entry_block: Block,
        args: list[Value],
        workload: list[Value],
        results: list[Value],
    ):
        self.entry_block = entry_block
        self.args = args
        self.workload = workload
        self.results = results

    @property
    def location(self) -> Location:
        return self.entry_block.owner.location

    def terminate(self):
        entry_block = self.entry_block
        with entry_block.owner.location, InsertionPoint(entry_block):
            func_d.ReturnOp([])
