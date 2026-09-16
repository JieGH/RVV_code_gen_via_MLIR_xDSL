	.file	"xdsl_api_gemm_8x8_b1_fp32.cpp"
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
	.globl	_Z33xdsl_api_gemm_RVV_8x8_b1_col_fp32PviPfS0_iS0_iS0_S0_i
	.type	_Z33xdsl_api_gemm_RVV_8x8_b1_col_fp32PviPfS0_iS0_iS0_S0_i, @function
_Z33xdsl_api_gemm_RVV_8x8_b1_col_fp32PviPfS0_iS0_iS0_S0_i:
.LFB1036:
	.cfi_startproc
	addi	sp,sp,-32	#,,
	.cfi_def_cfa_offset 32
	lla	t2,__stack_chk_guard	# tmp191,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:32:   float* v22 = &v8[v9];
	lw	a2,40(sp)		# v9, v9
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:19: void xdsl_api_gemm_RVV_8x8_b1_col_fp32(void*, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
	ld	a0, 0(t2)	# tmp197, __stack_chk_guard
	sd	a0, 8(sp)	# tmp197, D.92513
	li	a0, 0	# tmp197
	ld	t5,32(sp)		# v8, v8
	sd	ra,24(sp)	#,
	.cfi_offset 1, -8
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:31:   vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
	vsetivli	zero,8,e32,m1,ta,ma	#,,,,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:32:   float* v22 = &v8[v9];
	slli	a2,a2,2	#, _2, v9
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:32:   float* v22 = &v8[v9];
	add	t0,t5,a2	# _2, v22, v8
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:31:   vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v19);
	vle32.v	v8,0(t5)	# v42,* v8
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:35:   float* v25 = &v8[v24];
	add	t6,t0,a2	# _2, v25, v22
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:33:   vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v19);
	vle32.v	v31,0(t0)	# v43,* v22
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:38:   float* v28 = &v8[v27];
	add	t4,t6,a2	# _2, v28, v25
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:36:   vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v19);
	vle32.v	v30,0(t6)	# v44,* v25
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:41:   float* v31 = &v8[v30];
	add	t3,t4,a2	# _2, v31, v28
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:39:   vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v19);
	vle32.v	v29,0(t4)	# v45,* v28
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:44:   float* v34 = &v8[v33];
	add	t1,t3,a2	# _2, v34, v31
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:42:   vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v19);
	vle32.v	v28,0(t3)	# v46,* v31
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:47:   float* v37 = &v8[v36];
	add	a7,t1,a2	# _2, v37, v34
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:45:   vfloat32m1_t v35 = __riscv_vle32_v_f32m1(v34, v19);
	vle32.v	v27,0(t1)	# v47,* v34
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:50:   float* v40 = &v8[v39];
	add	a0,a7,a2	# _2, v40, v37
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:48:   vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v19);
	vle32.v	v26,0(a7)	# v48,* v37
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:51:   vfloat32m1_t v41 = __riscv_vle32_v_f32m1(v40, v19);
	vle32.v	v25,0(a0)	# v49,* v40
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:68:   for (size_t i50 = v11; i50 < v1; i50 += v12) {
	beq	a1,zero,.L2	#, v1,,
	slli	a4,a4,2	#, _145, v4
	slli	a6,a6,2	#, _141, v6
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:68:   for (size_t i50 = v11; i50 < v1; i50 += v12) {
	li	a2,0		# i50,
.L3:
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:83:     vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
	flw	fa3,0(a5)	# MEM[(float *)_139], MEM[(float *)_139]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:72:     vfloat32m1_t v54 = __riscv_vle32_v_f32m1(v53, v52);
	vle32.v	v24,0(a3)	# v54,* ivtmp.7
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:86:     vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
	flw	fa4,4(a5)	# MEM[(float *)_139 + 4B], MEM[(float *)_139 + 4B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:68:   for (size_t i50 = v11; i50 < v1; i50 += v12) {
	addi	a2,a2,1	#, i50, i50
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:89:     vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
	flw	fa5,8(a5)	# MEM[(float *)_139 + 8B], MEM[(float *)_139 + 8B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:68:   for (size_t i50 = v11; i50 < v1; i50 += v12) {
	add	a3,a3,a4	# _145, ivtmp.7, ivtmp.7
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:92:     vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
	flw	fa1,12(a5)	# MEM[(float *)_139 + 12B], MEM[(float *)_139 + 12B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:95:     vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
	flw	fa2,16(a5)	# MEM[(float *)_139 + 16B], MEM[(float *)_139 + 16B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:83:     vfloat32m1_t v65 = __riscv_vfmacc_vf_f32m1(v63, v64, v54, v52);
	vfmacc.vf	v8,fa3,v24	# v42, MEM[(float *)_139], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:86:     vfloat32m1_t v68 = __riscv_vfmacc_vf_f32m1(v66, v67, v54, v52);
	vfmacc.vf	v31,fa4,v24	# v43, MEM[(float *)_139 + 4B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:98:     vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
	flw	fa3,20(a5)	# MEM[(float *)_139 + 20B], MEM[(float *)_139 + 20B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:101:     vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
	flw	fa4,24(a5)	# MEM[(float *)_139 + 24B], MEM[(float *)_139 + 24B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:89:     vfloat32m1_t v71 = __riscv_vfmacc_vf_f32m1(v69, v70, v54, v52);
	vfmacc.vf	v30,fa5,v24	# v44, MEM[(float *)_139 + 8B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:104:     vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
	flw	fa5,28(a5)	# MEM[(float *)_139 + 28B], MEM[(float *)_139 + 28B]
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:92:     vfloat32m1_t v74 = __riscv_vfmacc_vf_f32m1(v72, v73, v54, v52);
	vfmacc.vf	v29,fa1,v24	# v45, MEM[(float *)_139 + 12B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:68:   for (size_t i50 = v11; i50 < v1; i50 += v12) {
	add	a5,a5,a6	# _141, ivtmp.8, ivtmp.8
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:95:     vfloat32m1_t v77 = __riscv_vfmacc_vf_f32m1(v75, v76, v54, v52);
	vfmacc.vf	v28,fa2,v24	# v46, MEM[(float *)_139 + 16B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:98:     vfloat32m1_t v80 = __riscv_vfmacc_vf_f32m1(v78, v79, v54, v52);
	vfmacc.vf	v27,fa3,v24	# v47, MEM[(float *)_139 + 20B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:101:     vfloat32m1_t v83 = __riscv_vfmacc_vf_f32m1(v81, v82, v54, v52);
	vfmacc.vf	v26,fa4,v24	# v48, MEM[(float *)_139 + 24B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:104:     vfloat32m1_t v86 = __riscv_vfmacc_vf_f32m1(v84, v85, v54, v52);
	vfmacc.vf	v25,fa5,v24	# v49, MEM[(float *)_139 + 28B], v54,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:68:   for (size_t i50 = v11; i50 < v1; i50 += v12) {
	bne	a2,a1,.L3	#, i50, v1,
.L2:
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:123:   __riscv_vse32_v_f32m1(v95, v87, v19);
	vse32.v	v8,0(t5)	# v42,* v8,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:125:   __riscv_vse32_v_f32m1(v96, v88, v19);
	vse32.v	v31,0(t0)	# v43,* v22,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:128:   __riscv_vse32_v_f32m1(v98, v89, v19);
	vse32.v	v30,0(t6)	# v44,* v25,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:131:   __riscv_vse32_v_f32m1(v100, v90, v19);
	vse32.v	v29,0(t4)	# v45,* v28,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:134:   __riscv_vse32_v_f32m1(v102, v91, v19);
	vse32.v	v28,0(t3)	# v46,* v31,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:137:   __riscv_vse32_v_f32m1(v104, v92, v19);
	vse32.v	v27,0(t1)	# v47,* v34,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:140:   __riscv_vse32_v_f32m1(v106, v93, v19);
	vse32.v	v26,0(a7)	# v48,* v37,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:143:   __riscv_vse32_v_f32m1(v108, v94, v19);
	vse32.v	v25,0(a0)	# v49,* v40,
# kernels/xdsl_api_gemm_8x8_b1_fp32.cpp:145: }
	ld	a4, 8(sp)	# tmp198, D.92513
	ld	a5, 0(t2)	# tmp190, __stack_chk_guard
	xor	a5, a4, a5	# tmp190, tmp198
	li	a4, 0	# tmp198
	bne	a5,zero,.L11	#, tmp190,,
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
	.size	_Z33xdsl_api_gemm_RVV_8x8_b1_col_fp32PviPfS0_iS0_iS0_S0_i, .-_Z33xdsl_api_gemm_RVV_8x8_b1_col_fp32PviPfS0_iS0_iS0_S0_i
	.ident	"GCC: (Bianbu 13.2.0-23ubuntu4bb3) 13.2.0"
	.section	.note.GNU-stack,"",@progbits
