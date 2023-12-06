from typing import Optional

from .ir import (
    Context,
    Location,
    Operation,
    builtin_d,
)


class ModuleBuilder:
    def __init__(
        self,
        *,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None
    ):
        if module_op:
            self.module_op = module_op
            self.body_block = module_op.regions[0].blocks[0]
        else:
            if not context:
                context = Context()
            self.module_op = builtin_d.ModuleOp(loc=Location.unknown(context))
            self.body_block = self.module_op.body
        self.context = self.module_op.context
