CXX = g++
CC = gcc

# Basic flags for RISC-V Vector extension
CFLAGS = -O3 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing
CXXFLAGS = -O3 -std=c++17 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing

# All generated files are C++ now
CXX_SRCS = $(wildcard *.cpp)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

TARGET = test_gemm.x

all: $(TARGET)

$(TARGET): $(CXX_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# No C files to compile

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)

DATASETS_PATTERN ?= ../../datasets/*.dat

run: $(TARGET)
	@for dataset in $(DATASETS_PATTERN); do \
		echo "Running dataset: $$dataset"; \
		./$(TARGET) $$dataset | tee "results_$$(basename $$dataset .dat).csv"; \
	done

check_vlen:
	@vlen=$$(cat /sys/devices/virtual/riscv_isa 2>/dev/null | grep -o 'v256\|v512\|v128' || echo "unknown"); \
	echo "Detected VLEN: $$vlen"; \
	# You can add logic here to assert vlen matches expected
