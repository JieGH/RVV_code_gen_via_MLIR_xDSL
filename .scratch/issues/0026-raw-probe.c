/* Issue #42 probe: do the X60 raw L2/prefetch events count? Streams 64 KiB (fits L2) and 4 MiB (misses L2), reads and writes. */
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static int op(uint32_t type, uint64_t cfg) {
  struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
  pe.size = sizeof pe; pe.type = type; pe.config = cfg; pe.exclude_kernel = 1; pe.exclude_hv = 1;
  return syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}
struct { const char* n; uint64_t c; } ev[] = {
  {"l1d_load_miss", 5}, {"l1d_access", 170}, {"l1d_miss", 171}, {"l1d_excl_evict", 172},
  {"l1d_prefetch_refill", 174}, {"l1d_prefetch_hit", 175},
  {"l2_load_access", 184}, {"l2_load_miss", 185}, {"l2_ar_channel_request", 186},
  {"l2_ar_channel_stall_cycle", 187}, {"l2_store_access", 188}, {"l2_aw_channel_stall_cycle", 189},
};
int main(void) {
  size_t sizes[] = {64 << 10, 4 << 20};
  for (int w = 0; w < 2; w++)
  for (int s = 0; s < 2; s++) {
    size_t n = sizes[s] / sizeof(float); float* a = calloc(n, sizeof(float)); volatile float sum = 0;
    for (size_t j = 0; j < n; j += 16) a[j] = 1;  /* touch */
    printf("== %s %zu KiB x 20\n", w ? "write" : "read", sizes[s] >> 10);
    for (unsigned i = 0; i < sizeof ev / sizeof ev[0]; i++) {
      int fd = op(PERF_TYPE_RAW, ev[i].c);
      if (fd < 0) { printf("  %-26s cannot open\n", ev[i].n); continue; }
      ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
      for (int r = 0; r < 20; r++)
        if (w) for (size_t j = 0; j < n; j += 16) a[j] = (float) r;
        else for (size_t j = 0; j < n; j += 16) sum += a[j];
      ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
      uint64_t v = 0; read(fd, &v, sizeof v); close(fd);
      printf("  %-26s %12llu  (%.3f per 64B line)\n", ev[i].n, (unsigned long long)v, v / (20.0 * sizes[s] / 64));
    }
    free(a);
  }
}
