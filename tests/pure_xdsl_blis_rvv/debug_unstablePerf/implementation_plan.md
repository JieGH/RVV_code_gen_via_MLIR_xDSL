# Implement RVV Packer Throttling (LMUL Reduction)

We will modify the xDSL code generation (`pack_builder.py`) to enforce `LMUL=1` for FP32 packing, breaking the massive 16-element memory bursts into smaller 8-element chunks to prevent the hardware L2 cache from panicking.

## Exact Code Modifications

### [MODIFY] [pack_builder.py](file:///home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/packing/pack_builder.py)

We will replace the inner memory generation logic for all four FP32 builders. Below is the exact Python MLIR generation code that will be used to replace the current logic in `build_pack_CB_v_if` (and similarly adapted for the other three builders).

**Current Logic:**
```python
    SetvlCls, _, LoadStridedCls, StoreCls = _lmul_ops(RR, elem_bytes, vlen_bits)
    vl_op = SetvlCls(nr.result)
    bstr = arith.MuliOp(ldM, cEB.result)
    ob.add_ops([rem, nr, vl_op, bstr])
    
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, c1.result, [], Region([ib])))
    i = ib.args[0]
    
    j_ldM = arith.MuliOp(j, ldM)
    src_off = arith.AddiOp(j_ldM.result, i)
    load_op = LoadStridedCls(M, src_off.result, bstr.result, vl_op.vl)
    
    j_mc = arith.MuliOp(j, mc)
    i_RR = arith.MuliOp(i, cRR.result)
    dst_off = arith.AddiOp(j_mc.result, i_RR.result)
    store = StoreCls(Mc, dst_off.result, load_op.result, vl_op.vl)
    
    ib.add_ops([j_ldM, src_off, load_op, j_mc, i_RR, dst_off, store, scf.YieldOp()])
```

**New Throttled Logic (LMUL Splitting):**
```python
    import math
    chunk_size = 8  # Enforce m1 vector length
    num_chunks = math.ceil(RR / chunk_size)
    
    # Force m1 classes manually to bypass dynamic scaling
    SetvlCls = SetvlOp
    LoadStridedCls = vlse32_v_f32m1Op
    StoreCls = vse32_v_f32m1Op
    
    bstr = arith.MuliOp(ldM, cEB.result)
    ob.add_ops([rem, nr, bstr])
    
    ib = Block(arg_types=[idx])
    ob.add_op(scf.ForOp(c0.result, mc, c1.result, [], Region([ib])))
    i = ib.args[0]
    
    ops_to_add = []
    current_rem = nr.result
    c8 = _const(chunk_size, idx)
    ops_to_add.append(c8)
    
    for chunk_idx in range(num_chunks):
        # 1. Safely calculate vector length for this chunk: max(current_rem, 0)
        rem_pos = arith.MaxSIOp(current_rem, c0.result)
        vl_k = arith.MinSIOp(rem_pos.result, c8.result)
        vl_op = SetvlCls(vl_k.result)
        
        # 2. Source offset: (j + chunk_idx*8) * ldM + i
        c_chunk = _const(chunk_idx * chunk_size, idx)
        j_plus_chunk = arith.AddiOp(j, c_chunk.result)
        j_ldM = arith.MuliOp(j_plus_chunk.result, ldM)
        src_off = arith.AddiOp(j_ldM.result, i)
        
        # 3. Load
        load_op = LoadStridedCls(M, src_off.result, bstr.result, vl_op.vl)
        
        # 4. Destination offset: (j * mc) + (i * RR) + chunk_idx*8
        j_mc = arith.MuliOp(j, mc)
        i_RR = arith.MuliOp(i, cRR.result)
        dst_base = arith.AddiOp(j_mc.result, i_RR.result)
        dst_off = arith.AddiOp(dst_base.result, c_chunk.result)
        
        # 5. Store
        store = StoreCls(Mc, dst_off.result, load_op.result, vl_op.vl)
        
        ops_to_add.extend([
            rem_pos, vl_k, vl_op, c_chunk, j_plus_chunk, j_ldM, src_off, 
            load_op, j_mc, i_RR, dst_base, dst_off, store
        ])
        
        # 6. Update remaining elements for the next chunk
        if chunk_idx < num_chunks - 1:
            next_rem = arith.SubiOp(current_rem, c8.result)
            ops_to_add.append(next_rem)
            current_rem = next_rem.result

    ops_to_add.append(scf.YieldOp())
    ib.add_ops(ops_to_add)
```
*(This same unrolling template will be applied to `pack_CB_v_else`, `pack_RB_v_if`, and `pack_RB_v_else`, adjusting the stride/offset math `src_off` according to each function's specific memory layout).*

## Why this works
By injecting this Python unrolling, we completely avoid using `m2` instructions. Instead of emitting one massive `vle32.v` (which requests 16 elements in a single clock cycle and crashes the Cortex-A55 memory controller), the MLIR will emit two separate `vlse32.v` instructions bounded to 8 elements each. The integer offset arithmetic generated between the loads will act as a natural delay, smoothing out the memory bandwidth and keeping the L2 cache stable.

## Verification Plan
1. Apply the exact code blocks above to `pack_builder.py`.
2. Generate the project C++ files.
3. Validate that `gemm_pack_xdsl.cpp` contains the unrolled `m1` logic instead of `m2`.
4. Run the benchmark to confirm the `B3_M7` median GOPs variance drops without altering `modelLevel.c`.
