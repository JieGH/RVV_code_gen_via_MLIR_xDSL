#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef F16
typedef _Float16 T;
#include "xdsl_ukernels_f16f16f32.h"
#define TABLE "table_f16f16f32.inc"
#else
typedef float T;
#include "xdsl_ukernels_fp32.h"
#define TABLE "table_fp32.inc"
#endif
typedef void (*kfn)(int,int,int,const T*,const T*,float*,float,int,int,int);
static struct { int mr, nr; const char *name; kfn f; } tab[] = {
#include TABLE
};
static uint64_t s = 0x9E3779B97F4A7C15ull ^ 0xC0FFEE1234ull;
static double rnd(void){ s ^= s<<13; s ^= s>>7; s ^= s<<17; return (s>>11)*(1.0/9007199254740992.0); }
#define G 64
enum { INT, RAND, RANDBIG };
static double gen(int mode){
  if(mode==INT) return (int)(rnd()*11)-5;
  if(mode==RAND) return 2*rnd()-1;
  return 2000*rnd()-1000;
}
static const char *mn[]={"int","rand","big1000"};
int main(void){
  int kcs[]={0,1,2,3,64,513,2000}, nk=7;
  int total=0, fails_k=0; long calls=0;
  int kc0_zero=0, kc0_bad=0, zsign=0;
  int nt=sizeof tab/sizeof tab[0];
  for(int t=0;t<nt;t++){
    int MR=tab[t].mr, NR=tab[t].nr, bad=0; char why[200]="";
    for(int mode=0;mode<3;mode++){
#ifndef F16
      if(mode==RANDBIG) continue;
#endif
      for(int ki=0;ki<nk;ki++){
        int kc=kcs[ki]; size_t na=(size_t)kc*MR, nb=(size_t)kc*NR;
        T *Ab=malloc((na+2*G)*sizeof(T)), *Bb=malloc((nb+2*G)*sizeof(T));
        memset(Ab,0xA5,(na+2*G)*sizeof(T)); memset(Bb,0xA5,(nb+2*G)*sizeof(T));
        T *A=Ab+G,*B=Bb+G;
        for(size_t i=0;i<na;i++) A[i]=(T)gen(mode);   /* for f16 this rounds to FP16; ref reads back the same value */
        for(size_t i=0;i<nb;i++) B[i]=(T)gen(mode);
        T *Abk=malloc((na+2*G)*sizeof(T)), *Bbk=malloc((nb+2*G)*sizeof(T));
        memcpy(Abk,Ab,(na+2*G)*sizeof(T)); memcpy(Bbk,Bb,(nb+2*G)*sizeof(T));
        double *P=calloc(MR*NR,8), *S=calloc(MR*NR,8);
        for(int k=0;k<kc;k++) for(int j=0;j<NR;j++) for(int i=0;i<MR;i++){
          double pr=(double)A[k*MR+i]*(double)B[k*NR+j]; P[j*MR+i]+=pr; S[j*MR+i]+=fabs(pr);}
        for(int ldi=0;ldi<2;ldi++){ int Clda=ldi?MR+17:MR;
         for(int bi=0;bi<3;bi++){ float beta=bi==0?0.f:(bi==1?1.f:0.5f);
          size_t nc=(size_t)Clda*NR+2*G;
          float *Cb=malloc(nc*4),*Cbk=malloc(nc*4),*C0=malloc((size_t)MR*NR*4);
          memset(Cb,0xA5,nc*4);
          for(int j=0;j<NR;j++){ for(int i=0;i<MR;i++){
            float v = beta==0 ? NAN : (float)gen(mode==INT?INT:RAND);
            C0[j*MR+i]= v; Cb[G+j*Clda+i]=v; } }
          memcpy(Cbk,Cb,nc*4);
          tab[t].f(MR,NR,kc,A,B,Cb+G,beta,MR,NR,Clda); calls++;
          if(memcmp(Ab,Abk,(na+2*G)*sizeof(T))||memcmp(Bb,Bbk,(nb+2*G)*sizeof(T))){bad++;snprintf(why,200,"A/B modified/guard written (%s kc=%d)",mn[mode],kc);}
          for(size_t x=0;x<nc;x++){ int in=0;
            if(x>=G && x<G+(size_t)Clda*NR){ size_t y=x-G; in=(int)(y%Clda)<MR; }
            if(!in && memcmp(&Cb[x],&Cbk[x],4)){bad++;snprintf(why,200,"write outside tile idx %ld (%s kc=%d Clda=%d beta=%g)",(long)x-(long)G,mn[mode],kc,Clda,beta);break;}}
          for(int j=0;j<NR;j++) for(int i=0;i<MR;i++){
            float got=Cb[G+j*Clda+i]; double base = beta==0?0.0:(double)C0[j*MR+i];
            double ref=base+P[j*MR+i];
            if(kc==0 && beta==0){ if(got==0.f) kc0_zero++; else {kc0_bad++; if(!bad) snprintf(why,200,"kc=0 beta=0 C not zeroed: %g (note)",got);} continue; }
            if(kc==0){ if(got!=(float)ref){bad++;snprintf(why,200,"kc=0 beta!=0 C changed");} continue;}
            if(mode==INT){ uint32_t a,b; float r=(float)ref; memcpy(&a,&got,4); memcpy(&b,&r,4);
              if(a!=b){ if(got==r) zsign++; else {bad++;snprintf(why,200,"int mismatch got %g ref %g (kc=%d Clda=%d beta=%g)",got,ref,kc,Clda,beta);} } }
            else { double den=S[j*MR+i]+fabs(base), tol=1e-5*sqrt((double)kc)*den;
              if(!(fabs(got-ref)<=tol)){bad++;snprintf(why,200,"%s err %g tol %g got %g ref %g (kc=%d Clda=%d beta=%g)",mn[mode],fabs(got-ref),tol,got,ref,kc,Clda,beta);} }
          }
          free(Cb);free(Cbk);free(C0);
        }}
        free(Ab);free(Bb);free(Abk);free(Bbk);free(P);free(S);
      }
    }
    total++; if(bad){fails_k++; printf("FAIL %s: %d bad; last: %s\n",tab[t].name,bad,why);} 
  }
  printf("%s: %d kernels, %ld calls, %d failed; kc0&beta0: %d cells zero, %d nonzero; +-0 sign diffs (int mode): %d -> %s\n",
#ifdef F16
   "f16f16f32",
#else
   "fp32",
#endif
   total,calls,fails_k,kc0_zero,kc0_bad,zsign,fails_k?"FAIL":"PASS");
  return fails_k!=0;
}
