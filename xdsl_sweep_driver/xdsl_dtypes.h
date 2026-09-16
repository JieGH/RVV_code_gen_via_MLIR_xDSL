#if defined(FP16)
  #define DTYPE _Float16
  #define A_TYPE _Float16
  #define B_TYPE _Float16
  #define C_TYPE _Float16
  #define ALPHA_TYPE _Float16
  #define BETA_TYPE _Float16
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_fp16
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_8x8_prefetch_fp16
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_prefetch_fp16
#elif defined(FP32)
  #define DTYPE float
  #define A_TYPE float
  #define B_TYPE float
  #define C_TYPE float
  #define ALPHA_TYPE float
  #define BETA_TYPE float
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_fp32
  #define gemm_microkernel_Cresident_neon_4x4_prefetch_unroll gemm_microkernel_Cresident_neon_4x4_prefetch_unroll_fp32
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_8x8_prefetch_fp32
  #define gemm_microkernel_Cresident_neon_8x12 gemm_microkernel_Cresident_neon_8x12
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_fp32
#elif defined(FP64)
  #define DTYPE double
  #define A_TYPE double
  #define B_TYPE double
  #define C_TYPE double
  #define ALPHA_TYPE double
  #define BETA_TYPE double
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_fp64
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_4x8_prefetch_fp64
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_prefetch_fp64
#elif defined(I8I8I32)
  #include <stdint.h>
  #define DTYPE int8_t
  #define A_TYPE int8_t
  #define B_TYPE int8_t
  #define C_TYPE int32_t
  #define ALPHA_TYPE float
  #define BETA_TYPE float
  #define gemm_microkernel_Cresident_neon_4x4_prefetch gemm_microkernel_Cresident_neon_4x4_prefetch_i8i8i32
  #define gemm_microkernel_Cresident_neon_8x8_prefetch gemm_microkernel_Cresident_neon_4x8_prefetch_i8i8i32
  #define gemm_microkernel_ABresident_neon_4x4         gemm_microkernel_ABresident_neon_4x4_prefetch_i8i8i32
#endif
