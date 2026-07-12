from xdsl.context import Context
from xdsl.dialects import arith, scf, builtin
from xdsl.ir import Block, Region
from xdsl.printer import Printer

ctx = Context()
ctx.load_dialect(arith.Arith)
ctx.load_dialect(scf.Scf)

b = Block()
c0 = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType()))
c1 = arith.ConstantOp(builtin.IntegerAttr(1, builtin.IndexType()))
c3 = arith.ConstantOp(builtin.IntegerAttr(3, builtin.IndexType()))
b.add_ops([c0, c1, c3])

body = Block(arg_types=[builtin.IndexType(), builtin.IndexType()])
body.add_op(scf.YieldOp(body.args[1]))

f = scf.ForOp(c0.result, c1.result, c3.result, [c0.result], Region([body]))
b.add_op(f)

p = Printer()
p.print(b)
