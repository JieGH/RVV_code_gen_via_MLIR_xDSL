/* Build from the repo root: gcc -O1 -Itests/pure_xdsl_blis_rvv/modelLevel .scratch/issues/0020-model-check.c tests/pure_xdsl_blis_rvv/modelLevel/model_level.c -lm */
/* Issue #35 step 5: nc from get_optim_mc_nc_kc (packed B sized for an L3 the board lacks)
   next to nc from the same formula with the board's real last-level cache, the 512 KiB L2,
   as 8-way (bananapi.h's WL2) and as 16-way (the other value bananapi.h has commented). */
#include "model_level.h"
struct S { const char* l; int m, n, k, mr, nr; };
int main(void) {
  struct S s[] = {
    {"B2",384,384,64,16,15},{"B3",64,384,384,16,15},
    {"Reuse8",128,384,384,16,15},{"Reuse12",192,384,384,16,15},{"Reuse24",384,384,384,16,15},
    {"B1",1024,384,1024,32,6},{"B5",1024,384,4096,32,6},{"B4",4096,384,1024,32,7},
    {"S1",1000,1000,1000,32,7},{"S2",2000,2000,2000,32,7},{"S3",3000,3000,3000,32,7},
    {"S4",4000,4000,4000,32,7},{"S5",5000,5000,5000,32,7}};
  printf("%-8s %5s %5s %5s %4s %4s | %4s %4s | %-18s | %-18s | %-18s\n","layer","M","N","K","MR","NR","kc","mc",
         "nc L3 (shipped)","nc L2 8-way","nc L2 16-way");
  for (unsigned i = 0; i < sizeof s / sizeof s[0]; ++i) {
    int mc, nc, kc; get_optim_mc_nc_kc(4, s[i].m, s[i].n, s[i].k, s[i].mr, s[i].nr, &mc, &nc, &kc);
    int raw3 = model_level(1, NL3, CL3, WL3, 4, kc, mc);
    int raw2 = model_level(1, NL2, CL2, WL2, 4, kc, mc);
    int raw2b = model_level(1, 512, CL2, 16, 4, kc, mc);
    char a[32], b[32], c[32];
    snprintf(a, 32, "%d (raw %d)", min(s[i].n, raw3), raw3);
    snprintf(b, 32, "%d (raw %d)", min(s[i].n, raw2), raw2);
    snprintf(c, 32, "%d (raw %d)", min(s[i].n, raw2b), raw2b);
    printf("%-8s %5d %5d %5d %4d %4d | %4d %4d | %-18s | %-18s | %-18s\n", s[i].l, s[i].m, s[i].n, s[i].k,
           s[i].mr, s[i].nr, kc, mc, a, b, c);
  }
}
