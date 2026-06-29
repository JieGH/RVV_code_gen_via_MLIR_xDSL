CXX    = riscv64-linux-gnu-g++
CFLAGS = -march=rv64gcv -mabi=lp64d -O3 -mcmodel=medany -I.

# Block sizes — must match what compile_packing.sh used for code generation
NR ?= 8
MR ?= 4

all: verify_packing

# Link: driver + reference + single generated master kernel
verify_packing: driver_pack_check.o ref_pack_CB.o gen/pack_v_fp32.o
	$(CXX) $(CFLAGS) $^ -o $@

# Compile unified master kernel
gen/pack_v_fp32.o: gen/pack_v_fp32.c
	$(CXX) $(CFLAGS) -DNR=$(NR) -DMR=$(MR) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f verify_packing *.o gen/*.o
