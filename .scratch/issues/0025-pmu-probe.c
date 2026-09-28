/* Issue 0025 probe. Run on jbpi 2026-09-27: gcc -O1 p.c -o p && taskset -c 4 ./p */
/* Which perf events open on the X60, and do they count? Streams over 4 MiB (misses L2) and 64 KiB (hits L2). */
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/ioctl.h>
static int op(uint32_t type, uint64_t cfg) {
  struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
  pe.size = sizeof pe; pe.type = type; pe.config = cfg; pe.exclude_kernel = 1; pe.exclude_hv = 1;
  return syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}
#define C(id, o, r) (PERF_COUNT_HW_CACHE_##id | (PERF_COUNT_HW_CACHE_OP_##o << 8) | (PERF_COUNT_HW_CACHE_RESULT_##r << 16))
struct { const char* n; uint32_t t; uint64_t c; } ev[] = {
  {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
  {"cache-references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
  {"cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
  {"LL-read-access", PERF_TYPE_HW_CACHE, C(LL, READ, ACCESS)},
  {"LL-read-miss", PERF_TYPE_HW_CACHE, C(LL, READ, MISS)},
  {"LL-write-access", PERF_TYPE_HW_CACHE, C(LL, WRITE, ACCESS)},
  {"LL-write-miss", PERF_TYPE_HW_CACHE, C(LL, WRITE, MISS)},
  {"LL-prefetch-access", PERF_TYPE_HW_CACHE, C(LL, PREFETCH, ACCESS)},
  {"LL-prefetch-miss", PERF_TYPE_HW_CACHE, C(LL, PREFETCH, MISS)},
  {"L1D-prefetch-access", PERF_TYPE_HW_CACHE, C(L1D, PREFETCH, ACCESS)},
  {"L1D-prefetch-miss", PERF_TYPE_HW_CACHE, C(L1D, PREFETCH, MISS)},
  {"L1D-read-miss", PERF_TYPE_HW_CACHE, C(L1D, READ, MISS)},
  {"bus-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES},
};
int main(void) {
  size_t sizes[] = {64 << 10, 4 << 20};
  for (int s = 0; s < 2; s++) {
    size_t n = sizes[s] / sizeof(float); float* a = calloc(n, sizeof(float)); volatile float sum = 0;
    printf("== stream %zu KiB x 20\n", sizes[s] >> 10);
    for (unsigned i = 0; i < sizeof ev / sizeof ev[0]; i++) {
      int fd = op(ev[i].t, ev[i].c);
      if (fd < 0) { printf("  %-20s cannot open\n", ev[i].n); continue; }
      ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
      for (int r = 0; r < 20; r++) for (size_t j = 0; j < n; j += 16) sum += a[j];
      ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
      uint64_t v = 0; read(fd, &v, sizeof v); close(fd);
      printf("  %-20s %12llu  (%.2f per 64B line)\n", ev[i].n, (unsigned long long)v, v / (20.0 * sizes[s] / 64));
    }
    free(a);
  }
}
