	.file	"xdsl_api_gemm_4x4_b1_i8i8i32.cpp"
	.option nopic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
# GNU C++17 (Bianbu 13.2.0-23ubuntu4bb3) version 13.2.0 (riscv64-linux-gnu)
#	compiled by GNU C version 13.2.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mabi=lp64d -mcmodel=medany -mtune=spacemit-x60 -misa-spec=20191213 -march=rv64imafdcv_zicsr_zifencei_zve32f_zve32x_zve64d_zve64f_zve64x_zvl128b_zvl32b_zvl64b -O3 -fstack-protector-strong
	.text
	.align	1
	.globl	_Z36xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32PviPfPaiS1_iS0_Pii
	.type	_Z36xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32PviPfPaiS1_iS0_Pii, @function
_Z36xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32PviPfPaiS1_iS0_Pii:
.LFB1036:
	.cfi_startproc
	addi	sp,sp,-32	#,,
	.cfi_def_cfa_offset 32
	lla	t2,__stack_chk_guard	# tmp178,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:37:   int32_t* v27 = &v8[v26];
	lw	a2,40(sp)		# v9, v9
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	sd	ra,24(sp)	#,
	.cfi_offset 1, -8
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	ld	t0,32(sp)		# v8, v8
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:34:   vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v14);
	vsetivli	zero,4,e16,mf2,ta,ma	#,,,,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:37:   int32_t* v27 = &v8[v26];
	slli	a2,a2,2	#, _2, v9
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:37:   int32_t* v27 = &v8[v26];
	add	t6,t0,a2	# _2, v27, v8
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:41:   int32_t* v31 = &v8[v30];
	add	t5,t6,a2	# _2, v31, v27
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:45:   int32_t* v35 = &v8[v34];
	add	t4,t5,a2	# _2, v35, v31
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	ld	a2, 0(t2)	# tmp184, __stack_chk_guard
	sd	a2, 8(sp)	# tmp184, D.92481
	li	a2, 0	# tmp184
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:34:   vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v14);
	vle32.v	v28,0(t0)	# v37,* v8
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:38:   vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v14);
	vle32.v	v27,0(t6)	# v38,* v27
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:42:   vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v14);
	vle32.v	v26,0(t5)	# v39,* v31
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:46:   vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v14);
	vle32.v	v25,0(t4)	# v40,* v35
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:55:   for (size_t i41 = v10; i41 < v1; i41 += v11) {
	beq	a1,zero,.L2	#, v1,,
.L3:
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:66:     int16_t v52 = (int16_t) v51;
	lb	t3,0(a5)		# v52, MEM[(int8_t *)_110]
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:55:   for (size_t i41 = v10; i41 < v1; i41 += v11) {
	addi	a2,a2,1	#, i41, i41
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:70:     int16_t v56 = (int16_t) v55;
	lb	t1,1(a5)		# v56, MEM[(int8_t *)_110 + 1B]
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:59:     vint8mf4_t v45 = __riscv_vle8_v_i8mf4(v44, v43);
	vle8.v	v29,0(a3)	# v45,* ivtmp.7
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:74:     int16_t v60 = (int16_t) v59;
	lb	a7,2(a5)		# v60, MEM[(int8_t *)_110 + 2B]
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:60:     vint16mf2_t v46 = __riscv_vsext_vf2_i16mf2(v45, v43);
	vsext.vf2	v24,v29	# v46, v45,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:78:     int16_t v64 = (int16_t) v63;
	lb	a0,3(a5)		# v64, MEM[(int8_t *)_110 + 3B]
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:55:   for (size_t i41 = v10; i41 < v1; i41 += v11) {
	add	a3,a3,a4	# v4, ivtmp.7, ivtmp.7
	add	a5,a5,a6	# v6, ivtmp.8, ivtmp.8
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:68:     vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v46, v43);
	vwmacc.vx	v28,t3,v24	# v37, v52, v46,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:72:     vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v46, v43);
	vwmacc.vx	v27,t1,v24	# v38, v56, v46,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:76:     vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v46, v43);
	vwmacc.vx	v26,a7,v24	# v39, v60, v46,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:80:     vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v46, v43);
	vwmacc.vx	v25,a0,v24	# v40, v64, v46,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:55:   for (size_t i41 = v10; i41 < v1; i41 += v11) {
	bne	a2,a1,.L3	#, i41, v1,
.L2:
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:91:   __riscv_vse32_v_i32m1(v71, v67, v14);
	vse32.v	v28,0(t0)	# v37,* v8,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:93:   __riscv_vse32_v_i32m1(v72, v68, v14);
	vse32.v	v27,0(t6)	# v38,* v27,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:96:   __riscv_vse32_v_i32m1(v74, v69, v14);
	vse32.v	v26,0(t5)	# v39,* v31,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:99:   __riscv_vse32_v_i32m1(v76, v70, v14);
	vse32.v	v25,0(t4)	# v40,* v35,
# kernels/xdsl_api_gemm_4x4_b1_i8i8i32.cpp:101: }
	ld	a4, 8(sp)	# tmp185, D.92481
	ld	a5, 0(t2)	# tmp177, __stack_chk_guard
	xor	a5, a4, a5	# tmp177, tmp185
	li	a4, 0	# tmp185
	bne	a5,zero,.L11	#, tmp177,,
	ld	ra,24(sp)		#,
	.cfi_remember_state
	.cfi_restore 1
	addi	sp,sp,32	#,,
	.cfi_def_cfa_offset 0
	jr	ra		#
.L11:
	.cfi_restore_state
	call	__stack_chk_fail		#
	.cfi_endproc
.LFE1036:
	.size	_Z36xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32PviPfPaiS1_iS0_Pii, .-_Z36xdsl_api_gemm_RVV_4x4_b1_col_i8i8i32PviPfPaiS1_iS0_Pii
	.ident	"GCC: (Bianbu 13.2.0-23ubuntu4bb3) 13.2.0"
	.section	.note.GNU-stack,"",@progbits
