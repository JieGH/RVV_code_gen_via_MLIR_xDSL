import sys

with open("tests/gemm_blis_xdsl/pack_builder.py", "r") as f:
    content = f.read()

# Replace signatures
content = content.replace("def build_pack_CB_v_if(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str\n):", "def build_pack_CB_v_if(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8\n):")
content = content.replace("def build_pack_CB_v_else(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str\n):", "def build_pack_CB_v_else(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8\n):")
content = content.replace("def build_pack_RB_v_if(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str\n):", "def build_pack_RB_v_if(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8\n):")
content = content.replace("def build_pack_RB_v_else(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str\n):", "def build_pack_RB_v_else(\n    idx: IndexType, memref_type, RR: int, elem_bytes: int, suffix: str, vlen: int = 8\n):")

content = content.replace("def build_pack_CB_module(\n    idx: IndexType, memref_type, RR: int = 8, elem_bytes: int = 4, suffix: str = \"fp32\"\n) -> ModuleOp:", "def build_pack_CB_module(\n    idx: IndexType, memref_type, RR: int = 8, elem_bytes: int = 4, suffix: str = \"fp32\", vlen: int = 8\n) -> ModuleOp:")
content = content.replace("build_pack_CB_v_if(idx, memref_type, RR, elem_bytes, suffix),", "build_pack_CB_v_if(idx, memref_type, RR, elem_bytes, suffix, vlen),")
content = content.replace("build_pack_CB_v_else(idx, memref_type, RR, elem_bytes, suffix),", "build_pack_CB_v_else(idx, memref_type, RR, elem_bytes, suffix, vlen),")

content = content.replace("def build_pack_module(\n    idx: IndexType,\n    memref_type,\n    NR: int = 8,\n    MR: int = 4,\n    elem_bytes: int = 4,\n    suffix: str = \"fp32\",\n) -> ModuleOp:", "def build_pack_module(\n    idx: IndexType,\n    memref_type,\n    NR: int = 8,\n    MR: int = 4,\n    elem_bytes: int = 4,\n    suffix: str = \"fp32\",\n    vlen: int = 8,\n) -> ModuleOp:")
content = content.replace("build_pack_CB_v_if(idx, memref_type, NR, elem_bytes, suffix),", "build_pack_CB_v_if(idx, memref_type, NR, elem_bytes, suffix, vlen),")
content = content.replace("build_pack_CB_v_else(idx, memref_type, NR, elem_bytes, suffix),", "build_pack_CB_v_else(idx, memref_type, NR, elem_bytes, suffix, vlen),")
content = content.replace("build_pack_RB_v_if(idx, memref_type, MR, elem_bytes, suffix),", "build_pack_RB_v_if(idx, memref_type, MR, elem_bytes, suffix, vlen),")
content = content.replace("build_pack_RB_v_else(idx, memref_type, MR, elem_bytes, suffix),", "build_pack_RB_v_else(idx, memref_type, MR, elem_bytes, suffix, vlen),")

# Now replace the bodies... Wait, this is too fragile with regex. Let's just rewrite the whole file using a template.
