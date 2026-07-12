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
from macro_builder import build_macro_module
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


def generate_header(config: BLISProjectConfig, out_dir: str, generated_variants: list):
    """Generates the C++ header for the generated kernel variants."""
    header_path = os.path.join(out_dir, "generated_kernels.h")
    with open(header_path, "w") as f:
        f.write("#ifndef GENERATED_KERNELS_H\n")
        f.write("#define GENERATED_KERNELS_H\n\n")
        f.write(f'#define CONFIG_DATATYPE "{config.datatype}"\n')
        f.write(f'#define CONFIG_METHOD "{config.method}"\n')
        f.write(f'#define CONFIG_VLEN {config.vlen}\n')
        f.write(f'#define CONFIG_ARCH "{config.arch}"\n\n')
        f.write('#ifdef __cplusplus\nextern "C" {\n#endif\n\n')

        for variant in generated_variants:
            f.write(
                f"void {variant['blis_name']}(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc);\n\n"
            )

        f.write(
            "typedef void (*gemm_blis_fn)(int m, int n, int k, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc, float* Ac, float* Bc, int mc, int nc, int kc);\n\n"
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
                f'    {{"{name_str}", {variant["mr"]}, {variant["nr"]}, {variant["blis_name"]}}},\n'
            )
        f.write("};\n\n")
        f.write(f"static const int num_variants = {len(generated_variants)};\n\n")

        f.write("#ifdef __cplusplus\n}\n#endif\n\n")
        f.write("#endif // GENERATED_KERNELS_H\n")


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

    # 1. Load config
    config = BLISProjectConfig.from_json(args.config)
    print(f"Loaded config: {args.config}")

    # Create project directory
    out_dir = f"pj_gen/{config.datatype}_{config.method}_vlen{config.vlen}"
    os.makedirs(out_dir, exist_ok=True)

    generated_variants = []

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

    # 2. Expand families and find global max
    max_mr = 0
    max_nr = 0
    expanded_families = set()
    for mr, nr in config.mr_nr_family:
        if mr > max_mr: max_mr = mr
        if nr > max_nr: max_nr = nr
        
        start_m = 2 if mr >= 2 else 1
        start_n = 2 if nr >= 2 else 1
        
        for i in range(start_m, mr + 1):
            for j in range(start_n, nr + 1):
                expanded_families.add((i, j))
    
    # Sort them descending
    expanded_families = sorted(list(expanded_families), key=lambda x: (x[0], x[1]), reverse=True)
    
    print(f"\n[+] Global Max MR={max_mr}, NR={max_nr}")
    print(f"[+] Expanded to {len(expanded_families)} independent macrokernels")

    vlen_elements = config.vlen // 32

    b0_modules = []
    b1_modules = []
    blis_modules = []
    pack_modules = []
    generated_microkernels_b0 = []
    generated_microkernels_b1 = []

    # 3. Generate Global Microkernel Pool
    print("\n[+] Generating Global Microkernel Pool...")
    for m_idx in range(1, max_mr + 1):
        for n_idx in range(1, max_nr + 1):
            # Beta 0
            mic_mod_b0 = build_gemm(
                mr=m_idx, nr=n_idx, vlen=vlen_elements, beta=BetaMode.B0, precision=config.datatype
            )
            default_name_b0 = f"xdsl_api_gemm_RVV_{m_idx}x{n_idx}_b0_col_{config.datatype}"
            new_name_b0 = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b0"
            rename_symbols(mic_mod_b0, {default_name_b0: new_name_b0})
            b0_modules.append(mic_mod_b0)
            generated_microkernels_b0.append((m_idx, n_idx, new_name_b0))

            # Beta 1
            mic_mod_b1 = build_gemm(
                mr=m_idx, nr=n_idx, vlen=vlen_elements, beta=BetaMode.B1, precision=config.datatype
            )
            default_name_b1 = f"xdsl_api_gemm_RVV_{m_idx}x{n_idx}_b1_col_{config.datatype}"
            new_name_b1 = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b1"
            rename_symbols(mic_mod_b1, {default_name_b1: new_name_b1})
            b1_modules.append(mic_mod_b1)
            generated_microkernels_b1.append((m_idx, n_idx, new_name_b1))

    # 4. Generate BLIS and Pack for each expanded family
    for mr, nr in expanded_families:
        print(f"    -> Generating BLIS/Pack for MR={mr}, NR={nr}")
        suffix = f"{config.datatype}_MR{mr}_NR{nr}"
        
        blis_mod = build_macro_module(
            MR=mr, NR=nr, orderA="C", transA="N", orderB="C", transB="N", orderC="C", suffix=config.datatype
        )
        default_blis_name = f"gemm_blis_v_CN_CN_C_{config.datatype}"
        
        f32 = Float32Type()
        mem_f32 = MemRefType(f32, [-1])
        idx = IndexType()
        pack_mod = build_pack_module(
            idx, mem_f32, NR=nr, MR=mr, elem_bytes=4, suffix=config.datatype
        )
        
        rename_map = {
            "gemm_microkernel_v_dispatch": f"gemm_microkernel_v_dispatch_{suffix}",
            default_blis_name: f"gemm_blis_v_{suffix}",
            f"pack_CB_v_if_{config.datatype}_RR{nr}": f"pack_CB_v_if_{suffix}",
            f"pack_CB_v_else_{config.datatype}_RR{nr}": f"pack_CB_v_else_{suffix}",
            f"pack_RB_v_if_{config.datatype}_RR{mr}": f"pack_RB_v_if_{suffix}",
            f"pack_RB_v_else_{config.datatype}_RR{mr}": f"pack_RB_v_else_{suffix}",
        }
        rename_symbols(blis_mod, rename_map)
        rename_symbols(pack_mod, rename_map)
        
        blis_modules.append(blis_mod)
        pack_modules.append(pack_mod)
        
        generated_variants.append({
            "mr": mr,
            "nr": nr,
            "blis_name": f"gemm_blis_v_{suffix}",
        })

    # 5. Generate Headers and Dispatcher
    print("\n[+] Generating headers and dispatcher...")
    macros = "#define GET_PTR(ptr, off) (ptr + off)\n#define GET_BETA(pc, beta) (pc == 0 ? beta : 1.0f)\n\n"
    fwd_decls = 'extern "C" {\n'
    
    # Global microkernel decls
    for m_idx, n_idx, new_name in generated_microkernels_b0:
        fwd_decls += f"void {new_name}(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);\n"
    for m_idx, n_idx, new_name in generated_microkernels_b1:
        fwd_decls += f"void {new_name}(size_t, float*, float*, size_t, float*, size_t, float*, float*, size_t);\n"

    dispatch_body = ""
    for mr, nr in expanded_families:
        suffix = f"{config.datatype}_MR{mr}_NR{nr}"
        fwd_decls += f"\nvoid pack_CB_v_if_{suffix}(float*, float*, size_t, size_t, size_t);\n"
        fwd_decls += f"void pack_CB_v_else_{suffix}(float*, float*, size_t, size_t, size_t);\n"
        fwd_decls += f"void pack_RB_v_if_{suffix}(float*, float*, size_t, size_t, size_t);\n"
        fwd_decls += f"void pack_RB_v_else_{suffix}(float*, float*, size_t, size_t, size_t);\n"
        
        fwd_decls += f"\nvoid gemm_microkernel_v_dispatch_{suffix}(size_t, size_t, size_t, float, float*, float*, float, float*, size_t);\n"
        fwd_decls += f"void gemm_microkernel_v_wrapper_{suffix}(int32_t, int32_t, int32_t, float, float*, float*, float, float*, int32_t);\n"
        
        disp = f"\nvoid gemm_microkernel_v_dispatch_{suffix}(size_t mr, size_t nr, size_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, size_t ldc) {{\n"
        disp += "    if (beta == 0.0f) {\n"
        for m_idx in range(1, mr + 1):
            for n_idx in range(1, nr + 1):
                new_name = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b0"
                disp += f"        if (mr == {m_idx} && nr == {n_idx}) {{ {new_name}((size_t)kc, &alpha, Ac, (size_t){mr}, Bc, (size_t){nr}, &beta, C, (size_t)ldc); return; }}\n"
        disp += f'        fprintf(stderr, "[GEMM DISPATCH ERROR] No B0 kernel for mr=%zu nr=%zu (max MR={mr} NR={nr})\\n", mr, nr); abort();\n'
        disp += "    } else {\n"
        for m_idx in range(1, mr + 1):
            for n_idx in range(1, nr + 1):
                new_name = f"gemm_microkernel_v_{config.datatype}_{m_idx}x{n_idx}_b1"
                disp += f"        if (mr == {m_idx} && nr == {n_idx}) {{ {new_name}((size_t)kc, &alpha, Ac, (size_t){mr}, Bc, (size_t){nr}, &beta, C, (size_t)ldc); return; }}\n"
        disp += f'        fprintf(stderr, "[GEMM DISPATCH ERROR] No B1 kernel for mr=%zu nr=%zu (max MR={mr} NR={nr})\\n", mr, nr); abort();\n'
        disp += "    }\n"
        disp += "}\n"
        
        disp += f"\nvoid gemm_microkernel_v_wrapper_{suffix}(int32_t mr, int32_t nr, int32_t kc, float alpha, float* Ac, float* Bc, float beta, float* C, int32_t ldc) {{\n"
        disp += f"    gemm_microkernel_v_dispatch_{suffix}((size_t)mr, (size_t)nr, (size_t)kc, alpha, Ac, Bc, beta, C, (size_t)ldc);\n"
        disp += "}\n"
        dispatch_body += disp
        
    fwd_decls += "\n}\n"
    
    headers_path = os.path.join(out_dir, "gemm_headers.h")
    with open(headers_path, "w") as f:
        f.write("#ifndef GEMM_HEADERS_H\n#define GEMM_HEADERS_H\n\n")
        f.write("#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <algorithm>\n#include <riscv_vector.h>\n\n")
        f.write(macros)
        f.write(fwd_decls)
        f.write("\n#endif // GEMM_HEADERS_H\n")
        
    dispatch_path = os.path.join(out_dir, "gemm_dispatch_xdsl.cpp")
    with open(dispatch_path, "w") as f:
        f.write('#include "gemm_headers.h"\n\nextern "C" {\n')
        f.write(dispatch_body)
        f.write("\n}\n")

    # 6. Merge, Pass, and Lower
    def lower_modules(modules, name):
        if not modules: return
        print(f"\n[+] Merging and applying passes for {name}...")
        merged_mod = merge_modules(*modules)
        for p in pass_pipeline:
            p.apply(ctx, merged_mod)
            
        mlir_path = os.path.join(out_dir, f"{name}.mlir")
        with open(mlir_path, "w") as f:
            Printer(stream=f, print_generic_format=True).print_op(merged_mod)
            
        c_path = os.path.join(out_dir, f"{name}.cpp")
        try:
            subprocess.run([args.mlir_translate, "-allow-unregistered-dialect", "-mlir-to-cpp", mlir_path, "-o", c_path], check=True)
            with open(c_path) as f:
                c_code = f.read()
            with open(c_path, "w") as f:
                f.write(f'#include "gemm_headers.h"\n\nextern "C" {{\n\n{c_code}\n}}\n')
            print(f"    -> Lowered to {c_path}")
        except subprocess.CalledProcessError as e:
            print(f"    -> Failed to lower to C: {e}")

    lower_modules(b0_modules, "gemm_b0_xdsl")
    lower_modules(b1_modules, "gemm_b1_xdsl")
    lower_modules(blis_modules, "gemm_blis_xdsl")
    lower_modules(pack_modules, "gemm_pack_xdsl")

    # Generate the C++ header
    generate_header(config, out_dir, generated_variants)
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

if __name__ == "__main__":
    main()
