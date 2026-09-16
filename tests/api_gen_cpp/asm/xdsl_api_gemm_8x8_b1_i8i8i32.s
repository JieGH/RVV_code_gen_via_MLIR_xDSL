	.file	"xdsl_api_gemm_8x8_b1_i8i8i32.cpp"
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
	.globl	_Z36xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32PviPfPaiS1_iS0_Pii
	.type	_Z36xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32PviPfPaiS1_iS0_Pii, @function
_Z36xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32PviPfPaiS1_iS0_Pii:
.LFB1036:
	.cfi_startproc
	addi	sp,sp,-64	#,,
	.cfi_def_cfa_offset 64
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:34:   vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
	vsetivli	zero,8,e16,mf2,ta,ma	#,,,,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:37:   int32_t* v27 = &v8[v26];
	lw	a2,72(sp)		# v9, v9
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	sd	s2,32(sp)	#,
	ld	t0,64(sp)		# v8, v8
	sd	s1,40(sp)	#,
	sd	s0,48(sp)	#,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:37:   int32_t* v27 = &v8[v26];
	slli	a2,a2,2	#, _2, v9
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	sd	s4,16(sp)	#,
	.cfi_offset 18, -32
	.cfi_offset 9, -24
	.cfi_offset 8, -16
	.cfi_offset 20, -48
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:37:   int32_t* v27 = &v8[v26];
	add	s2,t0,a2	# _2, v27, v8
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	sd	s3,24(sp)	#,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:41:   int32_t* v31 = &v8[v30];
	add	s1,s2,a2	# _2, v31, v27
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	sd	ra,56(sp)	#,
	.cfi_offset 19, -40
	.cfi_offset 1, -8
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:45:   int32_t* v35 = &v8[v34];
	add	s0,s1,a2	# _2, v35, v31
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	lla	s3,__stack_chk_guard	# tmp198,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:49:   int32_t* v39 = &v8[v38];
	add	t2,s0,a2	# _2, v39, v35
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	ld	a0, 0(s3)	# tmp204, __stack_chk_guard
	sd	a0, 8(sp)	# tmp204, D.92533
	li	a0, 0	# tmp204
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:53:   int32_t* v43 = &v8[v42];
	add	s4,t2,a2	# _2, v43, v39
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:34:   vint32m1_t v24 = __riscv_vle32_v_i32m1(v23, v18);
	vle32.v	v8,0(t0)	# v53,* v8
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:57:   int32_t* v47 = &v8[v46];
	add	t6,s4,a2	# _2, v47, v43
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:38:   vint32m1_t v28 = __riscv_vle32_v_i32m1(v27, v18);
	vle32.v	v31,0(s2)	# v54,* v27
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:61:   int32_t* v51 = &v8[v50];
	add	t5,t6,a2	# _2, v51, v47
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:42:   vint32m1_t v32 = __riscv_vle32_v_i32m1(v31, v18);
	vle32.v	v30,0(s1)	# v55,* v31
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:46:   vint32m1_t v36 = __riscv_vle32_v_i32m1(v35, v18);
	vle32.v	v29,0(s0)	# v56,* v35
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:50:   vint32m1_t v40 = __riscv_vle32_v_i32m1(v39, v18);
	vle32.v	v28,0(t2)	# v57,* v39
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:54:   vint32m1_t v44 = __riscv_vle32_v_i32m1(v43, v18);
	vle32.v	v27,0(s4)	# v58,* v43
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:58:   vint32m1_t v48 = __riscv_vle32_v_i32m1(v47, v18);
	vle32.v	v26,0(t6)	# v59,* v47
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:62:   vint32m1_t v52 = __riscv_vle32_v_i32m1(v51, v18);
	vle32.v	v25,0(t5)	# v60,* v51
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:79:   for (size_t i61 = v10; i61 < v1; i61 += v11) {
	beq	a1,zero,.L2	#, v1,,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:79:   for (size_t i61 = v10; i61 < v1; i61 += v11) {
	li	a2,0		# i61,
.L3:
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:94:     int16_t v76 = (int16_t) v75;
	lb	t1,0(a5)		# v76, MEM[(int8_t *)_198]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:83:     vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, v63);
	vle8.v	v9,0(a3)	# v65,* ivtmp.7
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:98:     int16_t v80 = (int16_t) v79;
	lb	a7,1(a5)		# v80, MEM[(int8_t *)_198 + 1B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:84:     vint16mf2_t v66 = __riscv_vsext_vf2_i16mf2(v65, v63);
	vsext.vf2	v24,v9	# v66, v65,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:102:     int16_t v84 = (int16_t) v83;
	lb	a0,2(a5)		# v84, MEM[(int8_t *)_198 + 2B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:79:   for (size_t i61 = v10; i61 < v1; i61 += v11) {
	addi	a2,a2,1	#, i61, i61
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:106:     int16_t v88 = (int16_t) v87;
	lb	t4,3(a5)		# v88, MEM[(int8_t *)_198 + 3B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:96:     vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v66, v63);
	vwmacc.vx	v8,t1,v24	# v53, v76, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:110:     int16_t v92 = (int16_t) v91;
	lb	t3,4(a5)		# v92, MEM[(int8_t *)_198 + 4B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:100:     vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v66, v63);
	vwmacc.vx	v31,a7,v24	# v54, v80, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:114:     int16_t v96 = (int16_t) v95;
	lb	t1,5(a5)		# v96, MEM[(int8_t *)_198 + 5B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:104:     vint32m1_t v86 = __riscv_vwmacc_vx_i32m1(v85, v84, v66, v63);
	vwmacc.vx	v30,a0,v24	# v55, v84, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:118:     int16_t v100 = (int16_t) v99;
	lb	a7,6(a5)		# v100, MEM[(int8_t *)_198 + 6B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:79:   for (size_t i61 = v10; i61 < v1; i61 += v11) {
	add	a3,a3,a4	# v4, ivtmp.7, ivtmp.7
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:122:     int16_t v104 = (int16_t) v103;
	lb	a0,7(a5)		# v104, MEM[(int8_t *)_198 + 7B]
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:108:     vint32m1_t v90 = __riscv_vwmacc_vx_i32m1(v89, v88, v66, v63);
	vwmacc.vx	v29,t4,v24	# v56, v88, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:79:   for (size_t i61 = v10; i61 < v1; i61 += v11) {
	add	a5,a5,a6	# v6, ivtmp.8, ivtmp.8
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:112:     vint32m1_t v94 = __riscv_vwmacc_vx_i32m1(v93, v92, v66, v63);
	vwmacc.vx	v28,t3,v24	# v57, v92, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:116:     vint32m1_t v98 = __riscv_vwmacc_vx_i32m1(v97, v96, v66, v63);
	vwmacc.vx	v27,t1,v24	# v58, v96, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:120:     vint32m1_t v102 = __riscv_vwmacc_vx_i32m1(v101, v100, v66, v63);
	vwmacc.vx	v26,a7,v24	# v59, v100, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:124:     vint32m1_t v106 = __riscv_vwmacc_vx_i32m1(v105, v104, v66, v63);
	vwmacc.vx	v25,a0,v24	# v60, v104, v66,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:79:   for (size_t i61 = v10; i61 < v1; i61 += v11) {
	bne	a2,a1,.L3	#, i61, v1,
.L2:
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:143:   __riscv_vse32_v_i32m1(v115, v107, v18);
	vse32.v	v8,0(t0)	# v53,* v8,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:145:   __riscv_vse32_v_i32m1(v116, v108, v18);
	vse32.v	v31,0(s2)	# v54,* v27,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:148:   __riscv_vse32_v_i32m1(v118, v109, v18);
	vse32.v	v30,0(s1)	# v55,* v31,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:151:   __riscv_vse32_v_i32m1(v120, v110, v18);
	vse32.v	v29,0(s0)	# v56,* v35,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:154:   __riscv_vse32_v_i32m1(v122, v111, v18);
	vse32.v	v28,0(t2)	# v57,* v39,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:157:   __riscv_vse32_v_i32m1(v124, v112, v18);
	vse32.v	v27,0(s4)	# v58,* v43,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:160:   __riscv_vse32_v_i32m1(v126, v113, v18);
	vse32.v	v26,0(t6)	# v59,* v47,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:163:   __riscv_vse32_v_i32m1(v128, v114, v18);
	vse32.v	v25,0(t5)	# v60,* v51,
# kernels/xdsl_api_gemm_8x8_b1_i8i8i32.cpp:165: }
	ld	a4, 8(sp)	# tmp205, D.92533
	ld	a5, 0(s3)	# tmp197, __stack_chk_guard
	xor	a5, a4, a5	# tmp197, tmp205
	li	a4, 0	# tmp205
	bne	a5,zero,.L11	#, tmp197,,
	ld	ra,56(sp)		#,
	.cfi_remember_state
	.cfi_restore 1
	ld	s0,48(sp)		#,
	.cfi_restore 8
	ld	s1,40(sp)		#,
	.cfi_restore 9
	ld	s2,32(sp)		#,
	.cfi_restore 18
	ld	s3,24(sp)		#,
	.cfi_restore 19
	ld	s4,16(sp)		#,
	.cfi_restore 20
	addi	sp,sp,64	#,,
	.cfi_def_cfa_offset 0
	jr	ra		#
.L11:
	.cfi_restore_state
	call	__stack_chk_fail		#
	.cfi_endproc
.LFE1036:
	.size	_Z36xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32PviPfPaiS1_iS0_Pii, .-_Z36xdsl_api_gemm_RVV_8x8_b1_col_i8i8i32PviPfPaiS1_iS0_Pii
	.ident	"GCC: (Bianbu 13.2.0-23ubuntu4bb3) 13.2.0"
	.section	.note.GNU-stack,"",@progbits
