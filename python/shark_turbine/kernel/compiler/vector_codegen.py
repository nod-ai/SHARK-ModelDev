from typing import Type, Optional, Sequence, Union, cast

from dataclasses import dataclass

import torch.fx as fx

from .._support.indexing import (
    Grid,
    IndexingContext,
    KernelBuffer,
    SymbolDef,
    _is_kernel_buffer_meta_derived,
)

from .builder import (
    ModuleBuilder,
)

from .ir import (
    FunctionType,
    IndexType,
    InsertionPoint,
    IrType,
    Location,
    Value,
    func_d,
)


ArgTypeUnion = Union[SymbolDef, Type[KernelBuffer]]


@dataclass
class ArgMeta:
    name: Optional[str] = None
    node: Optional[fx.Node] = None
    grid_index: Optional[int] = None


class Signature:
    """Represents a function signature.

    Signatures can carry:
      - Input, output and temporary KernelBuffers
      - SymbolDef

    For now, if we enounter any of these, we emit them in declaration order.
    We need a better convention than this (i.e. inputs, then outputs, them symbols, them temporaries).
    """

    def __init__(self):
        self.args: list[tuple[ArgMeta, ArgTypeUnion]] = []

    def add_kernel_buffer(
        self, kb: Type[KernelBuffer], *, meta: Optional[ArgMeta] = None
    ):
        self.args.append((meta if meta is not None else ArgMeta(), kb))

    def add_symbol(self, sym: SymbolDef, *, meta: Optional[ArgMeta] = None):
        self.args.append((meta if meta is not None else ArgMeta(), sym))

    @property
    def arg_metas(self) -> Sequence[ArgMeta]:
        return (meta for meta, _ in self.args)

    def as_function_type(self) -> FunctionType:
        idx_c = IndexingContext.current()

        def sym_to_dim_asm(s: SymbolDef) -> str:
            static_value = idx_c.get_static_value(s)
            return "?" if static_value is None else str(static_value)

        def as_mlir_type(t: ArgTypeUnion) -> FunctionType:
            if isinstance(t, SymbolDef):
                return IndexType.get()
            elif _is_kernel_buffer_meta_derived(t):
                kb_t = t  # type: KernelBuffer
                element_type_asm = kb_t.element_type.ir_type_asm()
                symbolic_shape = kb_t.symbolic_shape
                if symbolic_shape is not None:
                    shape_asm = "x".join(sym_to_dim_asm(s) for s in kb_t.symbolic_shape)
                    spec_asm = f"{shape_asm}x{element_type_asm}"
                else:
                    # Unranked. Not well supported, but for completeness.
                    spec_asm = element_type_asm
                memref_asm = f"memref<{spec_asm}>"
                return IrType.parse(memref_asm)

        inputs = [as_mlir_type(arg) for _, arg in self.args]
        return FunctionType.get(inputs, [])

    def add_grid(self, grid: Type[Grid]):
        assert grid.symbolic_shape, "code emission requires a symbolically shaped grid"
        for index, s in enumerate(grid.symbolic_shape):
            self.add_symbol(s, meta=ArgMeta(grid_index=index, name=f"grid{index}"))

    def add_from_graph_placeholders(self, graph: fx.Graph):
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            t = node.type
            meta = ArgMeta(name=node.target, node=node)
            if _is_kernel_buffer_meta_derived(t):
                self.add_kernel_buffer(t, meta=meta)
            elif issubclass(t, SymbolDef):
                self.add_symbol(t, meta=meta)

    def __repr__(self):
        parts = []
        for meta, arg in self.args:
            part = repr(arg)
            if meta.name:
                part = f"{meta.name}: {part}"
            parts.append(part)
        return f"Signature({', '.join(parts)})"


class ThreadEmitter:
    """Emits a 'thread function' as a `func` with a signature derived from the gm."""

    def __init__(self, mb: ModuleBuilder, grid: Grid, sig: Signature):
        self.nv_map: dict[fx.Node, Value] = {}
        self.grid_index_map: list[Value] = [None] * grid.rank

        # TODO: Infer a location from graph.
        with InsertionPoint(mb.body_block), Location.unknown():
            ftype = sig.as_function_type()
            func_op = func_d.FuncOp("kernel", ftype)
            self.func_op = func_op
            arg_locs = [
                (
                    Location.name(meta.name)
                    if meta.name is not None
                    else Location.unknown()
                )
                for meta in sig.arg_metas
            ]
            self.entry_block = func_op.add_entry_block(arg_locs)

            # Bind all inputs in the node-value map.
            for block_arg, meta in zip(self.entry_block.arguments, sig.arg_metas):
                assert (
                    meta.node or meta.grid_index is not None
                ), "expected all signature args to have an associated node or grid_index"
                if meta.node:
                    self.nv_map[meta.node] = block_arg
                elif meta.grid_index is not None:
                    self.grid_index_map[meta.grid_index] = block_arg
        self.ip = InsertionPoint(self.entry_block)

    def emit_node(self, node: fx.Node):
        ...

    def emit_graph(self, graph: fx.Graph):
        ...

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])
