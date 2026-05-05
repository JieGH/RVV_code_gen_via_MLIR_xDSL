# Makefile for remote verification of xDSL Macro-Kernel

CXX = riscv64-linux-gnu-g++
CC  = riscv64-linux-gnu-gcc
CFLAGS = -march=rv64gcv -mabi=lp64d -O3 -mcmodel=medany -I. -I./src
CXXFLAGS = $(CFLAGS) -std=c++11

TARGET_MOCK = test_macro
TARGET_HW   = test_macro_hw
TARGET_PACK = benchmark_packing
KPATH       = kernels

all: $(TARGET_MOCK) $(TARGET_HW) $(TARGET_PACK)

$(TARGET_MOCK): src/driver_mock_check.cpp gen/gemm_macro_v.c
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TARGET_HW): src/driver_hw_check.cpp gen/gemm_macro_v.c \
              $(KPATH)/kernels_RVV_8x8_fp32.c \
              $(KPATH)/exo_matrix_RVV_fp32.c
	$(CXX) $(CXXFLAGS) -I$(KPATH) $^ -o $@

$(TARGET_PACK): src/driver_pack_benchmark.cpp gen/gemm_macro_v.c
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET_MOCK) $(TARGET_HW) $(TARGET_PACK)
