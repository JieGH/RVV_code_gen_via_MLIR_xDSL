# Comparison of RVV MAC Strategies (i8i8i32)

This report analyzes three methods for achieving mixed-precision Multiply-Accumulate (MAC) on RISC-V Vector (RVV) hardware, focusing on performance and register pressure.

| Method | Instructions per KC loop | Speed (8x8) | Register Pressure |
| :--- | :--- | :--- | :--- |
| **Baseline (Zero-MAC)** | 2-3 (vwmacc + vwadd) | 1x (Baseline) | Low |
| **Direct Widen (vwmul)** | 2 (vwmul + vwadd) | ~1.3x | Low |
| **Widen-Once MAC** | 1 (vwmacc.vx) + 1 sext | **4.75x** | **High** |

---

## 1. Baseline: Zero-Initialized vwmacc
**Logic**: `vd = vwmacc(zero, A, B)` followed by `acc = vwadd(acc, vd)`.
*   **Drawback**: Extremely redundant. The hardware widens the A-vector from 8-bit to 16-bit for every single column in the B-matrix.
*   **Performance**: Poor (high instruction overhead).
*   **Spilling**: Excellent. Uses standard 8-bit registers for storage and only one temporary 16-bit register.

## 2. Improved: vwmul + vwadd
**Logic**: `temp = vwmul(A, B)` followed by `acc = vwadd(acc, temp)`.
*   **Drawback**: Still re-widens the A-vector for every column.
*   **Performance**: Slight improvement over baseline.
*   **Spilling**: Low. Standard register usage.

## 3. Deeply Optimized: Widen-Once MAC
**Logic**: `A_wide = vsext(A)` (outside column loop), then `acc = vwmacc.vx(acc, B_scalar, A_wide)`.

### **The Performance Secret**
By widening the A-tile to `i16` **once**, we move the expensive widening logic out of the inner loop. The subsequent `vwmacc.vx` instructions operate on 16-bit inputs and produce 32-bit outputs in a single step, perfectly matching the target accumulator precision.

### **The Drawbacks: Register Pressure & Spilling**
The fastest method has a significant architectural cost:

#### **A. Vector Register Bloat**
*   In the baseline, the A-tile is stored as `i8`. In a 512-bit register (VLEN=512), one register holds 64 elements.
*   In "Widen-Once", the A-tile is stored as `i16`. It now consumes **twice the register space** for the same number of elements.

#### **B. Register Spilling Risk**
*   **Accumulator Pressure**: For a large `MR x NR` tile (e.g., 32x8), you already need 32 registers to hold the `i32` accumulators (m1).
*   **Storage Pressure**: If you widen the A-tile as well, you need additional registers. On a 32-register machine, **spilling to memory becomes inevitable** if the tile size exceeds the register file capacity.
*   **LMUL Impact**: If you use `LMUL=2` for the accumulators to gain performance, you effectively halve the available register count, making the "Widen-Once" strategy even harder to fit without spilling.

### **Summary Table: Register Usage (32-register machine)**
| Microkernel | Accumulators (i32) | Widened A-Tile (i16) | Total Registers | Spilling Danger |
| :--- | :--- | :--- | :--- | :--- |
| **8 x 8** | 8 | 1 | ~9 | Safe |
| **16 x 8** | 16 | 2 | ~18 | Moderate |
| **32 x 8** | 32 | 4 | ~36 | **Critical (Must Spill)** |

> [!IMPORTANT]
> **Conclusion**: The Widen-Once MAC is the gold standard for performance, but it sets a hard limit on the maximum `MR` dimension you can generate before the cost of memory spills outweighs the instruction savings.
