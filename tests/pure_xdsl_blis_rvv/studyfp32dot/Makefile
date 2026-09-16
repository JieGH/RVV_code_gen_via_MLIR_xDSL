CC = gcc
CFLAGS = -O3 -DSCALAR_BASELINE_MAIN -Wall -Wextra -Wno-unused-parameter

# RVV flags for SpacemiT K1 / Banana Pi
RVV_FLAGS = -march=rv64gcv -mabi=lp64d

all: rvv_test int8_test fp32_asm_test

rvv_test: rvv_dot_gemm_fp32.c
	$(CC) $(CFLAGS) $(RVV_FLAGS) $< -lm -o $@

int8_test: int8_vmadot_study.c
	$(CC) $(CFLAGS) $(RVV_FLAGS) $< -lm -o $@

fp32_asm_test: fp32_asm_dot_study.c
	$(CC) $(CFLAGS) $(RVV_FLAGS) $< -lm -o $@

run: all
	./rvv_test
	./int8_test
	./fp32_asm_test

clean:
	rm -f rvv_test int8_test fp32_asm_test
