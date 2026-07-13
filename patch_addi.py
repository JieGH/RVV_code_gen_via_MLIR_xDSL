import re

with open("tests/gemm_blis_xdsl/pack_builder.py", "r") as f:
    content = f.read()

content = content.replace(
    "j_ldM, src_off_base = arith.MuliOp(j, ldM), arith.AddiOp(None, i)\n    src_off_base.operands = [j_ldM.result, i]",
    "j_ldM = arith.MuliOp(j, ldM)\n    src_off_base = arith.AddiOp(j_ldM.result, i)"
)

content = content.replace(
    "i_ldM, src_off_base = arith.MuliOp(i, ldM), arith.AddiOp(None, j)\n    src_off_base.operands = [i_ldM.result, j]",
    "i_ldM = arith.MuliOp(i, ldM)\n    src_off_base = arith.AddiOp(i_ldM.result, j)"
)

with open("tests/gemm_blis_xdsl/pack_builder.py", "w") as f:
    f.write(content)
