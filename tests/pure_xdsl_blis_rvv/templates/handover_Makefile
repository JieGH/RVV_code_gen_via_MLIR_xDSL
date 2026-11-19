# Issue #112 hand-over: build and run the microkernel check on a K1 board (RVV 1.0, VLEN 256, Zvfh).
# gcc-14: gcc 13 does not declare the FP16 widening intrinsics (__riscv_vfwmacc_vf_f32m1 with _Float16). make CC=... to change.
CC     = gcc-14
CFLAGS ?= -O3 -march=rv64gcv_zfh_zvfh -mabi=lp64d

SRC = test_xdsl_ukernels.c xdsl_ukernels_fp32.c xdsl_ukernels_f16f16f32.c ref_ukernels.c

test_xdsl_ukernels.x: $(SRC) xdsl_ukernels_fp32.h xdsl_ukernels_f16f16f32.h ref_ukernels.h
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

run: test_xdsl_ukernels.x
	./test_xdsl_ukernels.x

clean:
	rm -f test_xdsl_ukernels.x

.PHONY: run clean
