import argparse
import os
import shutil
import subprocess
import sys

# Add parent directories to Python path so we can import the old modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../gemm_blis_xdsl"))
)

from config import BLISProjectConfig
from pack_builder import build_pack_module
from xdsl.context import Context
from xdsl.dialects import arith, builtin, memref, scf
from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import (
    Float32Type,
    IndexType,
    MemRefType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.printer import Printer

from test_codeGeneration import BetaMode, build_gemm
from xdsltemplate.dialects.rvv import RVV
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.gemm_to_arith import GemmToArithPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
from xdsltemplate.transforms.rvv_to_emitc import RVVToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass


def merge_modules(*modules: ModuleOp) -> ModuleOp:
    merged_mod = ModuleOp([])
    for mod in modules:
        for op in list(mod.body.block.ops):
            if isinstance(op, func_dialect.FuncOp):
                op.detach()
                merged_mod.body.block.add_op(op)
    return merged_mod


def rename_symbols(mod: ModuleOp, rename_map: dict):
    """
    Renames all FuncOps and CallOps according to the rename_map.
    rename_map is { old_name_string : new_name_string }
    """

    def walk_and_rename(region):
        for block in region.blocks:
            for op in block.ops:
                if isinstance(op, func_dialect.CallOp):
                    callee_name = op.callee.string_value()
                    if callee_name in rename_map:
                        op.callee = SymbolRefAttr(rename_map[callee_name])
                elif op.name == "emitc.call_opaque":
                    if "callee" in op.attributes:
                        callee_name = op.attributes["callee"].data
                        if callee_name in rename_map:
                            from xdsl.dialects.builtin import StringAttr

                            op.attributes["callee"] = StringAttr(
                                rename_map[callee_name]
                            )
                    elif hasattr(op, "callee"):
                        callee_name = op.callee.data
                        if callee_name in rename_map:
                            new_name = rename_map[callee_name]
                            if callee_name == "gemm_microkernel_v":
                                new_name = new_name.replace(
                                    "gemm_microkernel_v_", "gemm_microkernel_v_wrapper_"
                                )
                            from xdsl.dialects.builtin import StringAttr

                            op.callee = StringAttr(new_name)
                # Recursively walk regions
                for r in op.regions:
                    walk_and_rename(r)

    for op in mod.body.block.ops:
        if isinstance(op, func_dialect.FuncOp):
            current_name = op.sym_name.data
            # Check if this function needs renaming
            if current_name in rename_map:
                op.sym_name = StringAttr(rename_map[current_name])
            # Rename any calls inside this function recursively
            walk_and_rename(op.body)


def generate_header(
    config: BLISProjectConfig,
    out_dir: str,
    generated_variants: list,
    max_dim: int,
    c_type_a: str,
    c_type_b: str,
    c_type_c: str,
    c_type_alpha: str,
    pack_dims=None,
):
    header_path = os.path.join(out_dir, "generated_kernels.h")
    with open(header_path, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <stddef.h>\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f'#define CONFIG_DATATYPE "{config.datatype}"\n')
        f.write(f'#define CONFIG_METHOD "{config.method}"\n')
        f.write(f"#define CONFIG_VLEN {config.vlen}\n")
        f.write(f'#define CONFIG_ARCH "{config.arch}"\n')
        f.write(f"#define CONFIG_ITERATIONS {config.iterations}\n\n")
        if config.datatype in ("int8", "i8i8i32"):
            f.write("#define DTYPE_INT8 1\n\n")

        if config.skip_ref_run:
            f.write("#define SKIP_REF_RUN 1\n\n")

        # Debug/mitigation knob (issue #1): per-shape nc override, keyed by label prefix.
        # Does not modify get_optim_mc_nc_kc or the general cache-blocking model.
        overrides = config.nc_overrides or {}
        f.write("struct NcOverrideEntry { const char* prefix; int nc; };\n")
        if overrides:
            entries = ", ".join(
                f'{{"{prefix}", {nc}}}' for prefix, nc in overrides.items()
            )
            f.write(
                f"static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = {{ {entries} }};\n"
            )
        else:
            f.write(
                'static const NcOverrideEntry CONFIG_NC_OVERRIDES[] = { {"", 0} };\n'
            )
        f.write(
            f"static const size_t CONFIG_NC_OVERRIDES_COUNT = {len(overrides)};\n\n"
        )

        f.write("#ifdef __cplusplus\n")
        f.write('extern "C" {\n')
        f.write("#endif\n\n")

        _pack_dims = pack_dims if pack_dims is not None else range(1, max_dim + 1)
        for dim in _pack_dims:
            f.write(
                f"void pack_CB_v_if_{config.method}_{config.datatype}_NR{dim}({c_type_b}*, {c_type_b}*, size_t, size_t, size_t);\n"
            )
            f.write(
                f"void pack_CB_v_else_{config.method}_{config.datatype}_NR{dim}({c_type_b}*, {c_type_b}*, size_t, size_t, size_t);\n"
            )
            f.write(
                f"void pack_RB_v_if_{config.method}_{config.datatype}_MR{dim}({c_type_a}*, {c_type_a}*, size_t, size_t, size_t);\n"
            )
            f.write(
                f"void pack_RB_v_else_{config.method}_{config.datatype}_MR{dim}({c_type_a}*, {c_type_a}*, size_t, size_t, size_t);\n"
            )
        f.write("\n")

        f.write(
            f"void gemm_blis_generic(int m, int n, int k, {c_type_alpha} alpha, {c_type_a}* A, int lda, {c_type_b}* B, int ldb, {c_type_alpha} beta, {c_type_c}* C, int ldc, {c_type_a}* Ac, {c_type_b}* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack);\n\n"
        )
        f.write(
            f"typedef void (*gemm_blis_fn)(int m, int n, int k, {c_type_alpha} alpha, {c_type_a}* A, int lda, {c_type_b}* B, int ldb, {c_type_alpha} beta, {c_type_c}* C, int ldc, {c_type_a}* Ac, {c_type_b}* Bc, int mc, int nc, int kc, int mr, int nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack);\n\n"
        )
        f.write("struct KernelVariant {\n")
        f.write("    const char* name;\n")
        f.write("    int mr;\n")
        f.write("    int nr;\n")
        f.write("    gemm_blis_fn func;\n")
        f.write("};\n\n")

        f.write("static const KernelVariant generated_variants[] = {\n")
        for variant in generated_variants:
            name_str = f"MR{variant['mr']}_NR{variant['nr']}"
            f.write(
                f'    {{"{name_str}", {variant["mr"]}, {variant["nr"]}, gemm_blis_generic}},\n'
            )
        f.write("};\n\n")
        f.write(f"static const int num_variants = {len(generated_variants)};\n\n")

        f.write("#ifdef __cplusplus\n")
        f.write("}\n")
        f.write("#endif\n")


def build_project(config: BLISProjectConfig, args):
    # Create project directory
    out_dir = config.out_dir()
    os.makedirs(out_dir, exist_ok=True)

    generated_variants = []

    # Setup type parameters
    if config.datatype == "fp32":
        elem_bytes = 4
        mlir_type = Float32Type()
        c_type_a = c_type_b = c_type_c = "float"
        c_type_alpha = "float"
    elif config.datatype in ["int8", "i8i8i32"]:
        elem_bytes = 1
        from xdsl.dialects.builtin import IntegerType

        mlir_type = IntegerType(8)
        c_type_a = c_type_b = "int8_t"
        c_type_c = "int32_t"
        c_type_alpha = "float"
    else:
        elem_bytes = 4
        mlir_type = Float32Type()
        c_type_a = c_type_b = c_type_c = "float"
        c_type_alpha = "float"

    # Setup xDSL Context
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func_dialect.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(RVV)

    pass_pipeline = [
        GemmToArithPass(),
        MemRefToEmitCPass(),
        ArithToEmitCPass(),
        MemrefLoadToEmitcPass(),
        MemrefStoreToEmitcPass(),
        SCFToEmitCPass(),
        RVVToEmitCPass(),
    ]

    # Use the is_pinned flag from the parsed configuration
    is_pinned = config.is_pinned
    # Always initialize so they're defined in both branches (linter safety)
    pinned_mr, pinned_nr = config.mr_nr_family[0] if is_pinned else (0, 0)

    # 2. Expand families and find global max
    max_mr = 0
    max_nr = 0
    expanded_families = set()
    for mr, nr in config.mr_nr_family:
        if mr > max_mr:
            max_mr = mr
        if nr > max_nr:
            max_nr = nr

        for i in range(1, mr + 1):
            for j in range(1, nr + 1):
                expanded_families.add((i, j))

    # Sort them descending
    expanded_families = sorted(
        list(expanded_families), key=lambda x: (x[0], x[1]), reverse=True
    )

    if is_pinned:
        pinned_mr, pinned_nr = config.mr_nr_family[0]
        print(
            f"\n[+] Pinned kernel mode: only generating MR={pinned_mr}, NR={pinned_nr}"
        )
    else:
        print(f"\n[+] Search mode: Global Max MR={max_mr}, NR={max_nr}")
        print(f"[+] Expanded to {len(expanded_families)} independent macrokernels")

    # NOTE: For INT8 i8mf4, capacity per fractional reg = vlen/(8*4) = vlen/32,
    # which coincidentally equals the FP32 M1 capacity. This is correct for i8mf4.
    # If LMUL is ever changed to M1 for INT8, use: vlen_elements = config.vlen // 8
    vlen_elements = config.vlen // 32

    b0_modules = []
    b1_modules = []
    pack_modules = []
    generated_microkernels_b0 = []
    generated_microkernels_b1 = []

    # 3. Generate Global Microkernel Pool
    print("\n[+] Generating Global Microkernel Pool...")
    _precision = (
        "i8i8i32" if config.datatype in ("int8", "i8i8i32") else config.datatype
    )

    # Pinned mode: only 1 kernel. Search mode: full pool for BLIS fringe dispatch.
    kernel_pairs = (
        [(pinned_mr, pinned_nr)]
        if is_pinned
        else [(m, n) for m in range(1, max_mr + 1) for n in range(1, max_nr + 1)]
    )

    for m_idx, n_idx in kernel_pairs:
        # Beta 0
        mic_mod_b0 = build_gemm(
            mr=m_idx,
            nr=n_idx,
            vlen=vlen_elements,
            beta=BetaMode.B0,
            precision=_precision,
            method=config.method,
        )
        default_name_b0 = f"xdsl_api_gemm_RVV_{m_idx}x{n_idx}_b0_col_{_precision}"
        new_name_b0 = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b0"
        rename_symbols(mic_mod_b0, {default_name_b0: new_name_b0})
        b0_modules.append(mic_mod_b0)
        generated_microkernels_b0.append((m_idx, n_idx, new_name_b0))

        # Beta 1
        mic_mod_b1 = build_gemm(
            mr=m_idx,
            nr=n_idx,
            vlen=vlen_elements,
            beta=BetaMode.B1,
            precision=_precision,
            method=config.method,
        )
        default_name_b1 = f"xdsl_api_gemm_RVV_{m_idx}x{n_idx}_b1_col_{_precision}"
        new_name_b1 = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b1"
        rename_symbols(mic_mod_b1, {default_name_b1: new_name_b1})
        b1_modules.append(mic_mod_b1)
        generated_microkernels_b1.append((m_idx, n_idx, new_name_b1))

    # 3.5 Generate Deduplicated Packing Modules
    print("    -> Generating deduplicated Packing modules")
    max_dim = max(max_mr, max_nr)
    mem_type = MemRefType(mlir_type, [-1])
    idx = IndexType()

    # Pinned mode: only generate pack funcs for the exact MR and NR needed.
    # Search mode: generate for all dims up to max_dim.
    pack_gen_dims = (
        sorted(set([pinned_mr, pinned_nr])) if is_pinned else range(1, max_dim + 1)
    )

    for dim in pack_gen_dims:
        pack_mod = build_pack_module(
            idx,
            mem_type,
            NR=dim,
            MR=dim,
            elem_bytes=elem_bytes,
            suffix=config.datatype,
            vlen=vlen_elements,
            method=config.method,
        )
        rename_map = {
            f"pack_CB_v_if_{config.method}_{config.datatype}_RR{dim}": f"pack_CB_v_if_{config.method}_{config.datatype}_NR{dim}",
            f"pack_CB_v_else_{config.method}_{config.datatype}_RR{dim}": f"pack_CB_v_else_{config.method}_{config.datatype}_NR{dim}",
            f"pack_RB_v_if_{config.method}_{config.datatype}_RR{dim}": f"pack_RB_v_if_{config.method}_{config.datatype}_MR{dim}",
            f"pack_RB_v_else_{config.method}_{config.datatype}_RR{dim}": f"pack_RB_v_else_{config.method}_{config.datatype}_MR{dim}",
        }
        rename_symbols(pack_mod, rename_map)
        pack_modules.append(pack_mod)

    # 4. Generate Generic BLIS config
    variants_to_test = [(pinned_mr, pinned_nr)] if is_pinned else expanded_families
    for mr, nr in variants_to_test:
        generated_variants.append(
            {
                "mr": mr,
                "nr": nr,
                "blis_name": "gemm_blis_generic",
            }
        )

    # 5. Generate Headers, Dispatcher and Generic BLIS
    print("\n[+] Generating headers, generic dispatcher, and generic BLIS...")
    macros = f"#define GET_PTR(ptr, off) (ptr + off)\n#define GET_BETA(pc, beta) (pc == 0 ? beta : ({c_type_alpha})1)\n\n"
    fwd_decls = 'extern "C" {\n'

    # Global microkernel decls
    for m_idx, n_idx, new_name in generated_microkernels_b0:
        fwd_decls += f"void {new_name}(size_t, {c_type_alpha}*, {c_type_a}*, size_t, {c_type_b}*, size_t, {c_type_alpha}*, {c_type_c}*, size_t);\n"
    for m_idx, n_idx, new_name in generated_microkernels_b1:
        fwd_decls += f"void {new_name}(size_t, {c_type_alpha}*, {c_type_a}*, size_t, {c_type_b}*, size_t, {c_type_alpha}*, {c_type_c}*, size_t);\n"

    for dim in range(1, max_dim + 1):
        fwd_decls += f"\nvoid pack_CB_v_if_{config.method}_{config.datatype}_NR{dim}({c_type_b}*, {c_type_b}*, size_t, size_t, size_t);\n"
        fwd_decls += f"void pack_CB_v_else_{config.method}_{config.datatype}_NR{dim}({c_type_b}*, {c_type_b}*, size_t, size_t, size_t);\n"
        fwd_decls += f"void pack_RB_v_if_{config.method}_{config.datatype}_MR{dim}({c_type_a}*, {c_type_a}*, size_t, size_t, size_t);\n"
        fwd_decls += f"void pack_RB_v_else_{config.method}_{config.datatype}_MR{dim}({c_type_a}*, {c_type_a}*, size_t, size_t, size_t);\n"

    fwd_decls += f"\nvoid gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, {c_type_alpha} alpha, {c_type_a}* Ac, {c_type_b}* Bc, {c_type_alpha} beta, {c_type_c}* C, size_t ldc, size_t max_mr, size_t max_nr);\n"
    fwd_decls += "\n}\n"

    headers_path = os.path.join(out_dir, "gemm_headers.h")
    with open(headers_path, "w") as f:
        f.write("#ifndef GEMM_HEADERS_H\n#define GEMM_HEADERS_H\n\n")
        f.write(
            "#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <string>\n#include <algorithm>\n#include <riscv_vector.h>\n#include <chrono>\n\n"
        )
        f.write(macros)
        f.write(fwd_decls)
        f.write("\n#endif // GEMM_HEADERS_H\n")

    if is_pinned:
        # Pinned mode: simple dispatcher — always calls the ONE kernel.
        b0_name = f"gemm_microkernel_v_{config.datatype}_{pinned_mr}x{pinned_nr}_b0"
        b1_name = f"gemm_microkernel_v_{config.datatype}_{pinned_mr}x{pinned_nr}_b1"
        dispatch_body = f"\nvoid gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, {c_type_alpha} alpha, {c_type_a}* Ac, {c_type_b}* Bc, {c_type_alpha} beta, {c_type_c}* C, size_t ldc, size_t max_mr, size_t max_nr) {{\n"
        dispatch_body += f"    if (beta == ({c_type_alpha})0) {{ {b0_name}(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); }}\n"
        dispatch_body += f"    else {{ {b1_name}(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); }}\n"
        dispatch_body += "}\n"
    else:
        # Search mode: full if-else table for all fringe kernels.
        dispatch_body = f"\nvoid gemm_microkernel_v_dispatch_generic(size_t cur_mr, size_t cur_nr, size_t kc, {c_type_alpha} alpha, {c_type_a}* Ac, {c_type_b}* Bc, {c_type_alpha} beta, {c_type_c}* C, size_t ldc, size_t max_mr, size_t max_nr) {{\n"
        dispatch_body += f"    if (beta == ({c_type_alpha})0) {{\n"
        for m_idx in range(1, max_mr + 1):
            for n_idx in range(1, max_nr + 1):
                new_name = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b0"
                dispatch_body += f"        if (cur_mr == {m_idx} && cur_nr == {n_idx}) {{ {new_name}(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }}\n"
        dispatch_body += '        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for cur_mr=%zu cur_nr=%zu\\n", cur_mr, cur_nr); abort();\n'
        dispatch_body += "    } else {\n"
        for m_idx in range(1, max_mr + 1):
            for n_idx in range(1, max_nr + 1):
                new_name = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b1"
                dispatch_body += f"        if (cur_mr == {m_idx} && cur_nr == {n_idx}) {{ {new_name}(kc, &alpha, Ac, max_mr, Bc, max_nr, &beta, C, ldc); return; }}\n"
        dispatch_body += '        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for cur_mr=%zu cur_nr=%zu\\n", cur_mr, cur_nr); abort();\n'
        dispatch_body += "    }\n"
        dispatch_body += "}\n"

    dispatch_body += f"""
static inline void pack_RB_scalar(int32_t mc, int32_t nc, {c_type_a}* M, int32_t ldM, {c_type_a}* Mc, int32_t RR) {{
    int32_t i, j, ii, k, rr;
    for (i = 0; i < mc; i += RR) {{
        k = i * nc;
        rr = std::min(mc - i, RR);
        for (j = 0; j < nc; j++) {{
            for (ii = 0; ii < rr; ii++) {{
                Mc[k] = M[(j) * ldM + (i + ii)];
                k++;
            }}
            k += (RR - rr);
        }}
    }}
}}

static inline void pack_CB_scalar(int32_t mc, int32_t nc, {c_type_b}* M, int32_t ldM, {c_type_b}* Mc, int32_t RR) {{
    int32_t i, j, jj, k, nr;
    for (j = 0; j < nc; j += RR) {{
        k = j * mc;
        nr = std::min(nc - j, RR);
        for (i = 0; i < mc; i++) {{
            for (jj = 0; jj < nr; jj++) {{
                Mc[k] = M[(j + jj) * ldM + (i)];
                k++;
            }}
            k += (RR - nr);
        }}
    }}
}}

void gemm_blis_generic(int32_t m, int32_t n, int32_t k, {c_type_alpha} alpha, {c_type_a}* A, int32_t lda, {c_type_b}* B, int32_t ldb, {c_type_alpha} beta, {c_type_c}* C, int32_t ldc, {c_type_a}* Ac, {c_type_b}* Bc, int32_t mc, int32_t nc, int32_t kc, int32_t mr, int32_t nr, double* pack_time, double* kernel_time, uint64_t* pack_bytes, bool use_scalar_pack) {{
"""
    if is_pinned:
        # Pinned mode: declare named pack functions directly — no table needed.
        pinned_cb_fn = f"pack_CB_v_if_{config.method}_{config.datatype}_NR{pinned_nr}"
        pinned_rb_fn = f"pack_RB_v_if_{config.method}_{config.datatype}_MR{pinned_mr}"
        dispatch_body += f"""
    typedef void (*pack_CB_fn)({c_type_b}*, {c_type_b}*, size_t, size_t, size_t);
    typedef void (*pack_RB_fn)({c_type_a}*, {c_type_a}*, size_t, size_t, size_t);
    const pack_CB_fn pack_CB_direct = {pinned_cb_fn};
    const pack_RB_fn pack_RB_direct = {pinned_rb_fn};
"""
    else:
        # Search mode: function pointer tables indexed by dimension.
        dispatch_body += f"""    typedef void (*pack_CB_fn)({c_type_b}*, {c_type_b}*, size_t, size_t, size_t);
    typedef void (*pack_RB_fn)({c_type_a}*, {c_type_a}*, size_t, size_t, size_t);

    static const pack_CB_fn pack_CB_funcs[] = {{
        nullptr,
"""
        for dim in range(1, max_dim + 1):
            dispatch_body += (
                f"        pack_CB_v_if_{config.method}_{config.datatype}_NR{dim},\n"
            )
        dispatch_body += """    };

    static const pack_RB_fn pack_RB_funcs[] = {
        nullptr,
"""
        for dim in range(1, max_dim + 1):
            dispatch_body += (
                f"        pack_RB_v_if_{config.method}_{config.datatype}_MR{dim},\n"
            )
        dispatch_body += """    };"""

    dispatch_body += f"""

    for (int32_t jc = 0; jc < n; jc += nc) {{
        int32_t nc_remain = n - jc;
        int32_t cur_nc = std::min(nc_remain, nc);
        for (int32_t pc = 0; pc < k; pc += kc) {{
            int32_t kc_remain = k - pc;
            int32_t cur_kc = std::min(kc_remain, kc);
            int32_t offset_B = jc * ldb + pc;
            {c_type_b}* b_ptr = GET_PTR(B, offset_B);
            int32_t kc_pad = cur_kc;
"""
    if "DOT" in config.method:
        dispatch_body += """
            kc_pad = ((cur_kc + 3) / 4) * 4;
"""
    # Pre-compute the pack call lines so we stay inside a single f-string below.
    pack_cb_call = (
        "                pack_CB_direct(b_ptr, Bc, cur_kc, cur_nc, ldb);"
        if is_pinned
        else "                pack_CB_funcs[nr](b_ptr, Bc, cur_kc, cur_nc, ldb);"
    )
    pack_rb_call = (
        "                    pack_RB_direct(a_ptr, Ac, cur_mc, cur_kc, lda);"
        if is_pinned
        else "                    pack_RB_funcs[mr](a_ptr, Ac, cur_mc, cur_kc, lda);"
    )

    # Pre-compute the inner kernel call block.
    # In pinned mode, fringe tiles (cur_mr<mr or cur_nr<nr) use a small stack temp
    # buffer of exactly mr*nr so the hardcoded kernel never writes beyond valid C.
    # Main tiles (cur_mr==mr, cur_nr==nr) call the kernel directly.
    # In search mode the dispatch table selects the right fringe kernel, no guard needed.
    if is_pinned:
        kernel_call_block = f"""\
                        if (cur_mr < mr || cur_nr < nr) {{
                            // Fringe tile: use a temp buffer to avoid OOB write by the {pinned_mr}x{pinned_nr} kernel
                            {c_type_c} tmp_C[{pinned_mr} * {pinned_nr}];
                            memset(tmp_C, 0, sizeof({c_type_c}) * {pinned_mr} * {pinned_nr});
                            if (beta_actual != ({c_type_alpha})0) {{
                                for (int _ii = 0; _ii < cur_mr; _ii++)
                                    for (int _jj = 0; _jj < cur_nr; _jj++)
                                        tmp_C[_jj * {pinned_mr} + _ii] = c_ptr[_jj * ldc + _ii];
                            }}
                            gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, kc_pad, alpha, a_packed, b_packed, beta_actual, tmp_C, {pinned_mr}, mr, nr);
                            for (int _ii = 0; _ii < cur_mr; _ii++)
                                for (int _jj = 0; _jj < cur_nr; _jj++)
                                    c_ptr[_jj * ldc + _ii] = tmp_C[_jj * {pinned_mr} + _ii];
                        }} else {{
                            gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, kc_pad, alpha, a_packed, b_packed, beta_actual, c_ptr, ldc, mr, nr);
                        }}"""
    else:
        kernel_call_block = "                        gemm_microkernel_v_dispatch_generic(cur_mr, cur_nr, kc_pad, alpha, a_packed, b_packed, beta_actual, c_ptr, ldc, mr, nr);"

    dispatch_body += f"""
            int32_t mc_pad = (mc + mr - 1) / mr * mr;
            int32_t nc_pad = (nc + nr - 1) / nr * nr;

            if (cur_nc % nr != 0) {{
                int32_t full_nc_elements = (cur_nc / nr) * nr * kc_pad;
                int32_t pad_nc_elements = nr * kc_pad;
                memset(Bc + full_nc_elements, 0, pad_nc_elements * sizeof({c_type_b}));
            }}

            auto t_pack_start_B = std::chrono::high_resolution_clock::now();
            if (use_scalar_pack) {{
                pack_CB_scalar(cur_kc, cur_nc, b_ptr, ldb, Bc, nr);
            }} else {{
{pack_cb_call}
            }}
            auto t_pack_end_B = std::chrono::high_resolution_clock::now();
            *pack_time += std::chrono::duration<double>(t_pack_end_B - t_pack_start_B).count();
            *pack_bytes += cur_kc * cur_nc * sizeof({c_type_b});

            {c_type_alpha} beta_actual = GET_BETA(pc, beta);
            for (int32_t ic = 0; ic < m; ic += mc) {{
                int32_t mc_remain = m - ic;
                int32_t cur_mc = std::min(mc_remain, mc);
                int32_t offset_A = pc * lda + ic;
                {c_type_a}* a_ptr = GET_PTR(A, offset_A);

                if (cur_mc % mr != 0) {{
                    int32_t full_mc_elements = (cur_mc / mr) * mr * kc_pad;
                    int32_t pad_mc_elements = mr * kc_pad;
                    memset(Ac + full_mc_elements, 0, pad_mc_elements * sizeof({c_type_a}));
                }}

                auto t_pack_start_A = std::chrono::high_resolution_clock::now();
                if (use_scalar_pack) {{
                    pack_RB_scalar(cur_mc, cur_kc, a_ptr, lda, Ac, mr);
                }} else {{
{pack_rb_call}
                }}
                auto t_pack_end_A = std::chrono::high_resolution_clock::now();
                *pack_time += std::chrono::duration<double>(t_pack_end_A - t_pack_start_A).count();
                *pack_bytes += cur_mc * cur_kc * sizeof({c_type_a});



                for (int32_t jr = 0; jr < cur_nc; jr += nr) {{
                    int32_t nr_remain = cur_nc - jr;
                    int32_t cur_nr = std::min(nr_remain, nr);
                    for (int32_t ir = 0; ir < cur_mc; ir += mr) {{
                        int32_t mr_remain = cur_mc - ir;
                        int32_t cur_mr = std::min(mr_remain, mr);
                        int32_t row_C = ic + ir;
                        int32_t col_C = jc + jr;
                        int32_t offset_C = col_C * ldc + row_C;
                        {c_type_c}* c_ptr = GET_PTR(C, offset_C);
                        {c_type_a}* a_packed = GET_PTR(Ac, ir * kc_pad);
                        {c_type_b}* b_packed = GET_PTR(Bc, jr * kc_pad);

                        auto t_kern_start = std::chrono::high_resolution_clock::now();
{kernel_call_block}
                        auto t_kern_end = std::chrono::high_resolution_clock::now();
                        *kernel_time += std::chrono::duration<double>(t_kern_end - t_kern_start).count();
                    }}
                }}
            }}
        }}
    }}
}}
"""

    dispatch_path = os.path.join(out_dir, "gemm_dispatch_xdsl.cpp")
    with open(dispatch_path, "w") as f:
        f.write(
            f'// CONFIG METHOD: {config.method}\n// DATATYPE: {config.datatype}\n#include "gemm_headers.h"\n\nextern "C" {{\n'
        )
        f.write(dispatch_body)
        f.write("\n}\n")

    # 6. Merge, Pass, and Lower
    def lower_modules(modules, name):
        if not modules:
            return
        print(f"\n[+] Merging and applying passes for {name}...")
        merged_mod = merge_modules(*modules)
        for p in pass_pipeline:
            p.apply(ctx, merged_mod)

        mlir_path = os.path.join(out_dir, f"{name}.mlir")
        with open(mlir_path, "w") as f:
            Printer(stream=f, print_generic_format=True).print_op(merged_mod)

        c_path = os.path.join(out_dir, f"{name}.cpp")
        try:
            subprocess.run(
                [
                    args.mlir_translate,
                    "-allow-unregistered-dialect",
                    "-mlir-to-cpp",
                    mlir_path,
                    "-o",
                    c_path,
                ],
                check=True,
            )
            with open(c_path) as f:
                c_code = f.read()
            with open(c_path, "w") as f:
                f.write(
                    f'// CONFIG METHOD: {config.method}\n// DATATYPE: {config.datatype}\n#include "gemm_headers.h"\n\nextern "C" {{\n\n{c_code}\n}}\n'
                )
            print(f"    -> Lowered to {c_path}")
        except subprocess.CalledProcessError as e:
            print(f"    -> Failed to lower to C: {e}")

    lower_modules(b0_modules, "gemm_b0_xdsl")
    lower_modules(b1_modules, "gemm_b1_xdsl")

    lower_modules(pack_modules, "gemm_pack_xdsl")

    # Generate the C++ header
    _pack_dims = sorted(set([pinned_mr, pinned_nr])) if is_pinned else None
    generate_header(
        config,
        out_dir,
        generated_variants,
        max_dim,
        c_type_a,
        c_type_b,
        c_type_c,
        c_type_alpha,
        pack_dims=_pack_dims,
    )
    print(f"\n[+] Generated header: {os.path.join(out_dir, 'generated_kernels.h')}")

    # 3. Copy static templates
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    shutil.copy(os.path.join(templates_dir, "driver_benchmark.cpp"), out_dir)

    # Process Makefile to optionally restrict datasets
    makefile_path = os.path.join(templates_dir, "Makefile")
    with open(makefile_path) as f:
        makefile_content = f.read()

    # The generated file is now split into multiple files, and the template Makefile
    # just compiles all *.cpp files, so we don't need to rename specific files.

    # Rename the TARGET binary to be unique
    target_name = f"test_gemm_{config.datatype}_{config.method}.x"
    makefile_content = makefile_content.replace(
        "TARGET = test_gemm.x", f"TARGET = {target_name}"
    )

    if config.datasets and len(config.datasets) > 0:
        datasets_str = " ".join([f"../../datasets/{d}" for d in config.datasets])
        makefile_content = makefile_content.replace(
            "DATASETS_PATTERN ?= ../../datasets/*.dat",
            f"DATASETS_PATTERN ?= {datasets_str}",
        )
        print(
            f"[+] Configured Makefile to run specific datasets: {', '.join(config.datasets)}"
        )

    with open(os.path.join(out_dir, "Makefile"), "w") as f:
        f.write(makefile_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--mlir-translate",
        type=str,
        default="/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate",
    )
    args = parser.parse_args()

    configs = BLISProjectConfig.from_json(args.config)
    print(f"Loaded config: {args.config} with {len(configs)} designs")

    # Clear old generated projects to avoid running stale configs
    import shutil

    shutil.rmtree("pj_gen", ignore_errors=True)

    ordered_dirs = []
    for config in configs:
        print("\n============================================================")
        print(
            f"Building project for {config.datatype}_{config.method}_vlen{config.vlen}..."
        )
        print("============================================================")
        build_project(config, args)
        ordered_dirs.append(os.path.basename(config.out_dir()))

    with open("pj_gen/compile_order.txt", "w") as f:
        for d in ordered_dirs:
            f.write(d + "\n")


if __name__ == "__main__":
    main()
