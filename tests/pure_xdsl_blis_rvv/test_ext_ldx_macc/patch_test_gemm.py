import re

with open('src/test_gemm.c', 'r') as f:
    content = f.read()

# Replace the argv parsing and loop with file parsing
# The original reads: mmin = atoi(argv[9]); ... kstep = atoi(argv[17]);
# Then later: for ( m=mmin; m<=mmax; m+=mstep ){ for ... }

# We will inject the file parsing logic
new_main_start = """
  char variant[20];
  DTYPE *A  = NULL, *B  = NULL, *C  = NULL, *Cg = NULL, *Ac = NULL, *Bc = NULL, *Cc = NULL;
  DTYPE alpha = 1.0, beta = 0.0;
  double t1, t2, time, tmin = 0.0, error, nrm, tmp, errorthd, flops, GFLOPS;
  size_t i, j, nreps, visual = 0, ldA, ldB, ldC;
  size_t m, n, k, mc, nc, kc; 
  
  errorthd = 1.0e-4;
  char orderA = 'C', orderB = 'C', orderC = 'C';
  char transA = 'N', transB = 'N';
  char test = 'T';

  sprintf(variant, "ALG+EXO");

  FILE *fp = fopen("../../datasets/testCase_repeat.dat", "r");
  if (!fp) { printf("Cannot open dataset\\n"); return 1; }
  
  printf("  Layer Name                 | Unroll | Median GOPs [min, p5, p95, max, iqr, s, CV%, mx/md] | Status\\n");
  printf("--------------------------------------------------------------------------------------------------------\\n");

  char line[256];
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\\n') continue;
    char label[128];
    int m_in, n_in, k_in;
    if (sscanf(line, "%s %d %d %d", label, &m_in, &n_in, &k_in) != 4) continue;
    
    m = 16; // microkernel M
    n = 15; // microkernel N
    k = k_in;

    int mc_tmp, nc_tmp, kc_tmp;
    get_optim_mc_nc_kc(sizeof(DTYPE), m, n, k, 16, 16, &mc_tmp, &nc_tmp, &kc_tmp);
    mc = mc_tmp; nc = nc_tmp; kc = kc_tmp;
    
    A  = (DTYPE *)my_aligned_alloc(32, m*k*sizeof(DTYPE), 1);   
    B  = (DTYPE *)my_aligned_alloc(32, k*n*sizeof(DTYPE), 1);   
    C  = (DTYPE *)my_aligned_alloc(32, m*n*sizeof(DTYPE), 1);
    Ac = (DTYPE *)my_aligned_alloc(32, (16+mc)*(16+kc)*sizeof(DTYPE), 1);
    Bc = (DTYPE *)my_aligned_alloc(32, (16+kc)*(16+nc)*sizeof(DTYPE), 1);
    Cc = (DTYPE *)my_aligned_alloc(32, (16+mc)*(16+nc)*sizeof(DTYPE), 1);
    Cg = (DTYPE *) malloc(m*n*sizeof(DTYPE));   
    
    ldA = m; ldB = k; ldC = m;
    generate_matrix( orderA, m, k, A, ldA );
    generate_matrix( orderB, k, n, B, ldB );
    generate_matrix( orderC, m, n, C, ldC );
    
    for ( j=0; j<n; j++ )
        for ( i=0; i<m; i++ ) 
          Cgcol(i,j)=Ccol(i,j);
          
    ukrFunction**** ukrmatrix = allocateMatrix();
    fillMatrix(ukrmatrix);
    
    int N_REPS = 1000;
    double rep_gops[1000];
    double sum_gops = 0.0;
    
    // Check Correctness first
    gemm_blis_B3A2C0( orderA, orderB, orderC, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, 
                      Ac, Bc, mc, nc, kc, NULL, NULL, NULL, ukrmatrix);
                      
    gemm( orderA, orderB, orderC, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, Cg, ldC );
    
    char status[10] = "PASS";
    for ( j=0; j<n; j++ ) {
      for ( i=0; i<m; i++ ) {
        if ( dabs( Ccol(i,j) - Cgcol(i,j) ) > errorthd ) {
            strcpy(status, "FAIL");
        }
      }
    }
    
    for (int r = 0; r < N_REPS; r++) {
        t1 = dclock();
        gemm_blis_B3A2C0( orderA, orderB, orderC, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, 
                          Ac, Bc, mc, nc, kc, NULL, NULL, NULL, ukrmatrix);
        t2 = dclock();
        double t = t2 - t1;
        double gops = (2.0 * m * n * k) / (t * 1e9);
        rep_gops[r] = gops;
        sum_gops += gops;
    }
    
    // Sort array
    for (int i=0; i<N_REPS-1; i++) {
        for (int j=0; j<N_REPS-i-1; j++) {
            if (rep_gops[j] > rep_gops[j+1]) {
                double temp = rep_gops[j];
                rep_gops[j] = rep_gops[j+1];
                rep_gops[j+1] = temp;
            }
        }
    }
    
    double min_g = rep_gops[0];
    double p5_g = rep_gops[(int)(0.05 * N_REPS)];
    double med_g = rep_gops[N_REPS/2];
    double p95_g = rep_gops[(int)(0.95 * N_REPS)];
    double max_g = rep_gops[N_REPS-1];
    double iqr_g = rep_gops[(int)(0.75 * N_REPS)] - rep_gops[(int)(0.25 * N_REPS)];
    
    double mean_g = sum_gops / N_REPS;
    double sq_sum = 0.0;
    for (int r=0; r<N_REPS; r++) sq_sum += (rep_gops[r] - mean_g) * (rep_gops[r] - mean_g);
    double std_g = sqrt(sq_sum / N_REPS);
    double cv_g = (med_g > 0) ? (std_g / med_g) * 100.0 : 0.0;
    double mxmd_g = (med_g > 0) ? (max_g / med_g) : 0.0;
    
    printf("▶ %-26s | 16x15 %5.2f KerGOPs [min:%.1f, p5:%.1f, p95:%.1f, max:%.1f, iqr:%.1f, s:%.1f, CV:%.0f%%, mx/md:%.2f] | %s\\n",
           label, med_g, min_g, p5_g, p95_g, max_g, iqr_g, std_g, cv_g, mxmd_g, (strcmp(status, "PASS") == 0 ? "✅ PASS" : "❌ FAIL"));
           
    my_aligned_free(A); my_aligned_free(B); my_aligned_free(C);
    my_aligned_free(Ac); my_aligned_free(Bc); my_aligned_free(Cc);
    free(Cg);
  }
  fclose(fp);
  return 0;
"""

# Replace the body of main
main_start = content.find("int main(int argc, char *argv[]) {")
if main_start != -1:
    body_start = content.find("{", main_start) + 1
    new_content = content[:body_start] + new_main_start + "\\n}\\n"
    with open('src/test_gemm.c', 'w') as f:
        f.write(new_content)
        
