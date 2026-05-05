	.file	"xdsl_api_gemm_8x8_b0_i8i8i32.cpp"
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
	.globl	_Z36xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32PviPfPaiS1_iS0_Pii
	.type	_Z36xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32PviPfPaiS1_iS0_Pii, @function
_Z36xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32PviPfPaiS1_iS0_Pii:
.LFB1036:
	.cfi_startproc
	addi	sp,sp,-32	#,,
	.cfi_def_cfa_offset 32
	lla	t6,__stack_chk_guard	# tmp199,
	sd	ra,24(sp)	#,
	.cfi_offset 1, -8
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	ld	a2, 0(t6)	# tmp205, __stack_chk_guard
	sd	a2, 8(sp)	# tmp205, D.92509
	li	a2, 0	# tmp205
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:31:   vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
	vsetivli	zero,8,e32,m1,ta,ma	#,,,,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:19: void xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32(void*, int v1, float* v2, int8_t* v3, int v4, int8_t* v5, int v6, float* v7, int32_t* v8, int v9) {
	ld	t5,32(sp)		# v8, v8
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:31:   vint32m1_t v21 = __riscv_vmv_v_x_i32m1(v20, v18);
	vmv.v.i	v25,0	# v29,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:54:   v36 = v28;
	vmv1r.v	v26,v25	# v29, v36
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:52:   v35 = v27;
	vmv1r.v	v27,v25	# v29, v35
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:50:   v34 = v26;
	vmv1r.v	v28,v25	# v29, v34
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:48:   v33 = v25;
	vmv1r.v	v29,v25	# v29, v33
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:46:   v32 = v24;
	vmv1r.v	v30,v25	# v29, v32
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:44:   v31 = v23;
	vmv1r.v	v31,v25	# v29, v31
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:42:   v30 = v22;
	vmv1r.v	v8,v25	# v29, v30
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:55:   for (size_t i37 = v10; i37 < v1; i37 += v11) {
	beq	a1,zero,.L2	#, v1,,
	vsetvli	zero,zero,e16,mf2,ta,ma	#,,,
.L3:
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:70:     int16_t v52 = (int16_t) v51;
	lb	t1,0(a5)		# v52, MEM[(int8_t *)_191]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:59:     vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v39);
	vle8.v	v9,0(a3)	# v41,* ivtmp.7
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:74:     int16_t v56 = (int16_t) v55;
	lb	a7,1(a5)		# v56, MEM[(int8_t *)_191 + 1B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:60:     vint16mf2_t v42 = __riscv_vsext_vf2_i16mf2(v41, v39);
	vsext.vf2	v24,v9	# v42, v41,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:78:     int16_t v60 = (int16_t) v59;
	lb	a0,2(a5)		# v60, MEM[(int8_t *)_191 + 2B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:55:   for (size_t i37 = v10; i37 < v1; i37 += v11) {
	addi	a2,a2,1	#, i37, i37
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:82:     int16_t v64 = (int16_t) v63;
	lb	t4,3(a5)		# v64, MEM[(int8_t *)_191 + 3B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:72:     vint32m1_t v54 = __riscv_vwmacc_vx_i32m1(v53, v52, v42, v39);
	vwmacc.vx	v25,t1,v24	# v29, v52, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:86:     int16_t v68 = (int16_t) v67;
	lb	t3,4(a5)		# v68, MEM[(int8_t *)_191 + 4B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:76:     vint32m1_t v58 = __riscv_vwmacc_vx_i32m1(v57, v56, v42, v39);
	vwmacc.vx	v8,a7,v24	# v30, v56, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:90:     int16_t v72 = (int16_t) v71;
	lb	t1,5(a5)		# v72, MEM[(int8_t *)_191 + 5B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:80:     vint32m1_t v62 = __riscv_vwmacc_vx_i32m1(v61, v60, v42, v39);
	vwmacc.vx	v31,a0,v24	# v31, v60, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:94:     int16_t v76 = (int16_t) v75;
	lb	a7,6(a5)		# v76, MEM[(int8_t *)_191 + 6B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:55:   for (size_t i37 = v10; i37 < v1; i37 += v11) {
	add	a3,a3,a4	# v4, ivtmp.7, ivtmp.7
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:98:     int16_t v80 = (int16_t) v79;
	lb	a0,7(a5)		# v80, MEM[(int8_t *)_191 + 7B]
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:84:     vint32m1_t v66 = __riscv_vwmacc_vx_i32m1(v65, v64, v42, v39);
	vwmacc.vx	v30,t4,v24	# v32, v64, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:55:   for (size_t i37 = v10; i37 < v1; i37 += v11) {
	add	a5,a5,a6	# v6, ivtmp.8, ivtmp.8
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:88:     vint32m1_t v70 = __riscv_vwmacc_vx_i32m1(v69, v68, v42, v39);
	vwmacc.vx	v29,t3,v24	# v33, v68, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:92:     vint32m1_t v74 = __riscv_vwmacc_vx_i32m1(v73, v72, v42, v39);
	vwmacc.vx	v28,t1,v24	# v34, v72, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:96:     vint32m1_t v78 = __riscv_vwmacc_vx_i32m1(v77, v76, v42, v39);
	vwmacc.vx	v27,a7,v24	# v35, v76, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:100:     vint32m1_t v82 = __riscv_vwmacc_vx_i32m1(v81, v80, v42, v39);
	vwmacc.vx	v26,a0,v24	# v36, v80, v42,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:55:   for (size_t i37 = v10; i37 < v1; i37 += v11) {
	bne	a2,a1,.L3	#, i37, v1,
	vsetvli	zero,zero,e32,m1,ta,ma	#,,,
.L2:
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:120:   int32_t* v92 = &v8[v9];
	lw	a5,40(sp)		# v9, v9
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:119:   __riscv_vse32_v_i32m1(v91, v83, v18);
	vse32.v	v25,0(t5)	# v29,* v8,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:120:   int32_t* v92 = &v8[v9];
	slli	a5,a5,2	#, _13, v9
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:120:   int32_t* v92 = &v8[v9];
	add	t5,t5,a5	# _13, v92, v8
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:123:   int32_t* v94 = &v8[v93];
	add	a4,t5,a5	# _13, v94, v92
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:121:   __riscv_vse32_v_i32m1(v92, v84, v18);
	vse32.v	v8,0(t5)	# v30,* v92,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:126:   int32_t* v96 = &v8[v95];
	add	a3,a4,a5	# _13, v96, v94
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:124:   __riscv_vse32_v_i32m1(v94, v85, v18);
	vse32.v	v31,0(a4)	# v31,* v94,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:129:   int32_t* v98 = &v8[v97];
	add	a4,a3,a5	# _13, v98, v96
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:127:   __riscv_vse32_v_i32m1(v96, v86, v18);
	vse32.v	v30,0(a3)	# v32,* v96,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:132:   int32_t* v100 = &v8[v99];
	add	a3,a4,a5	# _13, v100, v98
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:130:   __riscv_vse32_v_i32m1(v98, v87, v18);
	vse32.v	v29,0(a4)	# v33,* v98,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:135:   int32_t* v102 = &v8[v101];
	add	a4,a3,a5	# _13, v102, v100
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:133:   __riscv_vse32_v_i32m1(v100, v88, v18);
	vse32.v	v28,0(a3)	# v34,* v100,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:138:   int32_t* v104 = &v8[v103];
	add	a5,a4,a5	# _13, v104, v102
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:136:   __riscv_vse32_v_i32m1(v102, v89, v18);
	vse32.v	v27,0(a4)	# v35,* v102,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:139:   __riscv_vse32_v_i32m1(v104, v90, v18);
	vse32.v	v26,0(a5)	# v36,* v104,
# kernels/xdsl_api_gemm_8x8_b0_i8i8i32.cpp:141: }
	ld	a4, 8(sp)	# tmp206, D.92509
	ld	a5, 0(t6)	# tmp198, __stack_chk_guard
	xor	a5, a4, a5	# tmp198, tmp206
	li	a4, 0	# tmp206
	bne	a5,zero,.L9	#, tmp198,,
	ld	ra,24(sp)		#,
	.cfi_remember_state
	.cfi_restore 1
	addi	sp,sp,32	#,,
	.cfi_def_cfa_offset 0
	jr	ra		#
.L9:
	.cfi_restore_state
	call	__stack_chk_fail		#
	.cfi_endproc
.LFE1036:
	.size	_Z36xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32PviPfPaiS1_iS0_Pii, .-_Z36xdsl_api_gemm_RVV_8x8_b0_col_i8i8i32PviPfPaiS1_iS0_Pii
	.ident	"GCC: (Bianbu 13.2.0-23ubuntu4bb3) 13.2.0"
	.section	.note.GNU-stack,"",@progbits
