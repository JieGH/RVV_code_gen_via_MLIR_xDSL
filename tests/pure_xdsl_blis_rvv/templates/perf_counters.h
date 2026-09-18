#ifndef PERF_COUNTERS_H
#define PERF_COUNTERS_H

// Hardware perf counters via a single perf_event_open() group (issue #8). Supplements,
// never replaces, the wall-clock pack/compute timing.

#include <linux/perf_event.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

// Issue #14: instrumentation switch.
//
// Every timed region costs a pair of clock reads, and the per-tile pair costs ~125-160
// ns on jbpi -- roughly half of it lands inside Compute_Time_ms and half falls between
// tiles, where no timer is running and so it is charged to no column at all. Building
// with -DRVV_INSTRUMENTED=0 compiles the clock pairs and the counter reads out
// entirely, leaving only the harness's one wall-clock read per rep: the same shape as
// the reference design's single gettimeofday. That build reports no pack/compute
// breakdown and no counters -- use the default build for those.
#ifndef RVV_INSTRUMENTED
#define RVV_INSTRUMENTED 1
#endif

#define PERF_NUM_EVENTS 10

enum {
    PERF_EV_CYCLES = 0,
    PERF_EV_INSTRUCTIONS,
    PERF_EV_L1D_READ,
    PERF_EV_L1D_READ_MISS,
    PERF_EV_L1D_WRITE,
    PERF_EV_L1D_WRITE_MISS,
    PERF_EV_DTLB_READ_MISS,
    PERF_EV_STALLED_BACKEND,
    PERF_EV_STALLED_FRONTEND,
    PERF_EV_BRANCH_MISSES
};

typedef struct {
    uint64_t v[PERF_NUM_EVENTS];
    uint64_t time_enabled;
    uint64_t time_running;
} perf_counts;

// Decodes a PERF_FORMAT_GROUP|TOTAL_TIME_ENABLED|TOTAL_TIME_RUNNING read.
static inline int perf_decode_group_read(const uint64_t* buf, long nbytes, perf_counts* out) {
    if (nbytes != (long) ((3 + PERF_NUM_EVENTS) * sizeof(uint64_t))) return -1;
    if (buf[0] != PERF_NUM_EVENTS) return -1;
    out->time_enabled = buf[1];
    out->time_running = buf[2];
    for (int i = 0; i < PERF_NUM_EVENTS; i++) out->v[i] = buf[3 + i];
    return 0;
}

typedef struct {
    double cycles;
    double instructions;
    double ipc;
    double l1d_read_miss_pct;
    double l1d_write_miss_pct;
    double dtlb_misses;
    double stall_be_pct;  // of cycles
    double stall_fe_pct;
    double branch_misses;
} perf_metrics;

static inline double perf_ratio(double num, double den) {
    return den > 0 ? num / den : NAN;
}

static inline perf_metrics perf_metrics_per_rep(const perf_counts* c, int reps) {
    const uint64_t* v = c->v;
    double cycles = (double) v[PERF_EV_CYCLES];
    perf_metrics m;
    m.cycles = cycles / reps;
    m.instructions = (double) v[PERF_EV_INSTRUCTIONS] / reps;
    m.ipc = perf_ratio((double) v[PERF_EV_INSTRUCTIONS], cycles);
    m.l1d_read_miss_pct = 100.0 * perf_ratio((double) v[PERF_EV_L1D_READ_MISS], (double) v[PERF_EV_L1D_READ]);
    m.l1d_write_miss_pct = 100.0 * perf_ratio((double) v[PERF_EV_L1D_WRITE_MISS], (double) v[PERF_EV_L1D_WRITE]);
    m.dtlb_misses = (double) v[PERF_EV_DTLB_READ_MISS] / reps;
    m.stall_be_pct = 100.0 * perf_ratio((double) v[PERF_EV_STALLED_BACKEND], cycles);
    m.stall_fe_pct = 100.0 * perf_ratio((double) v[PERF_EV_STALLED_FRONTEND], cycles);
    m.branch_misses = (double) v[PERF_EV_BRANCH_MISSES] / reps;
    return m;
}

// Rejects a window the group was descheduled for (multiplexed): its unscaled counts
// would be undercounts. Judged per window so one bad window can't poison later ones.
static inline int perf_counts_add_delta(perf_counts* region, const perf_counts* before,
                                        const perf_counts* after) {
    if (after->time_enabled - before->time_enabled != after->time_running - before->time_running) return -1;
    for (int i = 0; i < PERF_NUM_EVENTS; i++) region->v[i] += after->v[i] - before->v[i];
    return 0;
}

// One always-running counter group for the whole process. `valid` drops to 0 if the
// group could not be opened or any read in the current measurement was rejected.
typedef struct {
    int fd;
    int member_fds[PERF_NUM_EVENTS];
    int valid;
    perf_counts pack;
    perf_counts compute;
} perf_gemm_counters;

static inline void perf_gemm_close(perf_gemm_counters* p) {
    for (int i = 0; i < PERF_NUM_EVENTS; i++) {
        if (p->member_fds[i] >= 0) close(p->member_fds[i]);
        p->member_fds[i] = -1;
    }
    p->fd = -1;
    p->valid = 0;
}

static inline void perf_gemm_open(perf_gemm_counters* p) {
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < PERF_NUM_EVENTS; i++) p->member_fds[i] = -1;
    p->fd = -1;

#if RVV_INSTRUMENTED
#define PERF_CACHE_EV(id, op, res) \
    ((PERF_COUNT_HW_CACHE_##id) | ((PERF_COUNT_HW_CACHE_OP_##op) << 8) | ((PERF_COUNT_HW_CACHE_RESULT_##res) << 16))
    static const struct { uint32_t type; uint64_t config; } events[PERF_NUM_EVENTS] = {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {PERF_TYPE_HW_CACHE, PERF_CACHE_EV(L1D, READ, ACCESS)},
        {PERF_TYPE_HW_CACHE, PERF_CACHE_EV(L1D, READ, MISS)},
        {PERF_TYPE_HW_CACHE, PERF_CACHE_EV(L1D, WRITE, ACCESS)},
        {PERF_TYPE_HW_CACHE, PERF_CACHE_EV(L1D, WRITE, MISS)},
        {PERF_TYPE_HW_CACHE, PERF_CACHE_EV(DTLB, READ, MISS)},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
    };
#undef PERF_CACHE_EV

    for (int i = 0; i < PERF_NUM_EVENTS; i++) {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type = events[i].type;
        pe.size = sizeof(pe);
        pe.config = events[i].config;
        pe.disabled = (i == 0);
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        int fd = (int) syscall(__NR_perf_event_open, &pe, 0, -1, i == 0 ? -1 : p->member_fds[0], 0);
        if (fd < 0) {
            perf_gemm_close(p);
            return;
        }
        p->member_fds[i] = fd;
    }
    p->fd = p->member_fds[0];
    if (ioctl(p->fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
        ioctl(p->fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        perf_gemm_close(p);
        return;
    }
    p->valid = 1;
#endif  // RVV_INSTRUMENTED
}

static inline void perf_gemm_reset(perf_gemm_counters* p) {
    memset(&p->pack, 0, sizeof(p->pack));
    memset(&p->compute, 0, sizeof(p->compute));
    p->valid = (p->fd >= 0);
}

static inline void perf_snapshot(perf_gemm_counters* p, perf_counts* out) {
    uint64_t buf[3 + PERF_NUM_EVENTS];
    long got = (long) read(p->fd, buf, sizeof(buf));
    if (perf_decode_group_read(buf, got, out) != 0) p->valid = 0;
}

static inline void perf_region_begin(perf_gemm_counters* p, perf_counts* snap) {
#if RVV_INSTRUMENTED
    if (p->valid) perf_snapshot(p, snap);
#else
    (void) p;
    (void) snap;
#endif
}

static inline void perf_region_end(perf_gemm_counters* p, const perf_counts* snap, perf_counts* region) {
#if RVV_INSTRUMENTED
    if (!p->valid) return;
    // Zeroed because perf_snapshot leaves it untouched on a rejected read; the
    // `p->valid &&` below then skips it, but only the initialiser makes that local.
    perf_counts now = {};
    perf_snapshot(p, &now);
    if (p->valid && perf_counts_add_delta(region, snap, &now) != 0) p->valid = 0;
#else
    (void) p;
    (void) snap;
    (void) region;
#endif
}

#ifdef __cplusplus

#include <chrono>

// Scoped wall-clock timer: adds its own lifetime, in seconds, to *sink. Under
// RVV_INSTRUMENTED=0 the object holds nothing and the constructor does nothing, so
// both clock reads are gone rather than merely branched around -- which is the point,
// since the tax being measured away IS the pair of reads.
class rvv_region_timer {
#if RVV_INSTRUMENTED
    std::chrono::high_resolution_clock::time_point start_;
    double* sink_;

  public:
    explicit rvv_region_timer(double* sink)
        : start_(std::chrono::high_resolution_clock::now()), sink_(sink) {}
    ~rvv_region_timer() {
        *sink_ += std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - start_)
                      .count();
    }
#else
  public:
    explicit rvv_region_timer(double*) {}
#endif
    rvv_region_timer(const rvv_region_timer&) = delete;
    rvv_region_timer& operator=(const rvv_region_timer&) = delete;
};

// Scoped counter window: adds the group's delta over its lifetime to *region, skipping
// any window the group was descheduled for. Construct it OUTSIDE the rvv_region_timer
// it encloses, so its read() syscalls stay out of the wall-clock window.
class rvv_perf_region {
#if RVV_INSTRUMENTED
    perf_gemm_counters* p_;
    perf_counts* region_;
    perf_counts snap_{};

  public:
    rvv_perf_region(perf_gemm_counters* p, perf_counts* region) : p_(p), region_(region) {
        if (p_) perf_region_begin(p_, &snap_);
    }
    ~rvv_perf_region() {
        if (p_) perf_region_end(p_, &snap_, region_);
    }
#else
  public:
    rvv_perf_region(perf_gemm_counters*, perf_counts*) {}
#endif
    rvv_perf_region(const rvv_perf_region&) = delete;
    rvv_perf_region& operator=(const rvv_perf_region&) = delete;
};

#endif  // __cplusplus

#endif // PERF_COUNTERS_H
