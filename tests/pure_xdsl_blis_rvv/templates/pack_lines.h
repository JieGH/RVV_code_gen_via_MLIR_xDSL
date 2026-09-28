// Issue #43: clear pack-buffer lines with cbo.zero before the packer writes them.
//
// On the X60 a store to a line that is not in the cache first reads the line from the L2,
// even when the packer then overwrites all 64 bytes of it (#42's probe: ~1.0 l2_load_access
// per line written). cbo.zero (Zicboz) puts a line of zeros in the cache without that read.
// Checked on jbpi: it runs in user mode and riscv,cboz-block-size = 64.
//
// Only lines lying wholly inside the range are cleared; a partly written line at either end
// is left alone, so nothing outside the range is touched. On the host the helper does nothing
// (the packers' plain stores still write every slot), unless a test builds with
// -DRVV_ZERO_LINES_EMULATE, which clears the same lines with memset.
#ifndef RVV_PACK_LINES_H
#define RVV_PACK_LINES_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RVV_CBOZ_LINE 64

// The whole lines inside [p, p + bytes): `count` lines starting at address `first`.
struct rvv_line_span {
    uintptr_t first;
    size_t count;
};

static inline rvv_line_span rvv_whole_lines(const void* p, size_t bytes) {
    const uintptr_t start = (uintptr_t) p, end = start + bytes;
    const uintptr_t first = (start + RVV_CBOZ_LINE - 1) & ~(uintptr_t) (RVV_CBOZ_LINE - 1);
    const uintptr_t last = end & ~(uintptr_t) (RVV_CBOZ_LINE - 1);
    return {first, last > first ? (size_t) ((last - first) / RVV_CBOZ_LINE) : 0};
}

static inline void rvv_zero_whole_lines(void* p, size_t bytes) {
    const rvv_line_span s = rvv_whole_lines(p, bytes);
#if defined(RVV_ZERO_LINES_EMULATE)
    memset((void*) s.first, 0, s.count * RVV_CBOZ_LINE);
#elif defined(__riscv)
    // cbo.zero 0(rs1), spelled as its encoding (MISC-MEM, funct3 2, imm 4) so the build
    // does not need a -march with Zicboz.
    for (size_t i = 0; i < s.count; ++i)
        __asm__ volatile(".insn i 0x0F, 2, x0, %0, 4" : : "r"(s.first + i * RVV_CBOZ_LINE) : "memory");
#else
    (void) s;
#endif
}

#endif  // RVV_PACK_LINES_H
