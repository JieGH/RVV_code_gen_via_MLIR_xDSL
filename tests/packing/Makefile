CXX    = riscv64-linux-gnu-g++
CFLAGS = -march=rv64gcv -mabi=lp64d -O3 -mcmodel=medany -I.

# Block sizes — single source of truth, must match compile_packing.sh
# These are passed as -DNR/-DMR to every translation unit so that
# driver_pack_check.cpp and benchmark_packing.cpp pick them up
# via the #ifndef guards instead of hardcoded constants.
NR ?= 8
MR ?= 8

DEFINES = -DNR=$(NR) -DMR=$(MR)

DTYPE ?= fp32
METHOD ?= AXPY_FP32_Pack

ifeq ($(DTYPE), i8)
DEFINES += -DTEST_INT8
endif

all: verify_packing benchmark_packing

# Link: driver + reference + single generated master kernel
verify_packing: driver_pack_check.o ref_pack_CB.o ref_pack_DOT_INT8_LANE_Pack.o gen/pack_v_$(METHOD)_$(DTYPE).o
	$(CXX) $(CFLAGS) $^ -o $@

# Link: benchmark driver + reference + single generated master kernel
benchmark_packing: benchmark_packing.o ref_pack_CB.o ref_pack_DOT_INT8_LANE_Pack.o gen/pack_v_$(METHOD)_$(DTYPE).o
	$(CXX) $(CFLAGS) $^ -o $@

# Compile unified master kernel (block sizes baked in)
gen/pack_v_$(METHOD)_$(DTYPE).o: gen/pack_v_$(METHOD)_$(DTYPE).c
	$(CXX) $(CFLAGS) $(DEFINES) -c $< -o $@

# Compile all C++ sources with NR/MR defines so driver & benchmark
# can use #ifndef NR / #ifndef MR instead of hardcoded values
%.o: %.cpp
	$(CXX) $(CFLAGS) $(DEFINES) -c $< -o $@

clean:
	rm -f verify_packing benchmark_packing *.o gen/*.o
