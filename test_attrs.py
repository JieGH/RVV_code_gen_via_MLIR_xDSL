from xdsl.ir import Operation, Block, Region
from xdsl.dialects.builtin import StringAttr, ModuleOp
from xdsl.dialects import func
from xdsl.parser import Parser
from xdsl.context import MLContext
from xdsl.dialects.emitc import Emitc
ctx = MLContext()
ctx.load_dialect(Emitc)
mod = ModuleOp([])
print(mod.attributes)
