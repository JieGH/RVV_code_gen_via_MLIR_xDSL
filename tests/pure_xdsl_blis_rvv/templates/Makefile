CXX = g++
CC = gcc

# Basic flags for RISC-V Vector extension
CFLAGS = -O3 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing
CXXFLAGS = -O2 -std=c++17 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing -fno-var-tracking-assignments --param ggc-min-expand=1 --param ggc-min-heapsize=32768

# All generated files are C++ now, plus model_level.c
CXX_SRCS = $(wildcard *.cpp)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

C_SRCS = ../../modelLevel/model_level.c
C_OBJS = $(C_SRCS:.c=.o)

TARGET = test_gemm.x

all: $(TARGET)

$(TARGET): $(CXX_OBJS) $(C_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)

DATASETS_PATTERN ?= ../../datasets/*.dat

run: $(TARGET)
	@mkdir -p logs
	@for dataset in $(DATASETS_PATTERN); do \
		echo "Running dataset: $$dataset"; \
		./$(TARGET) $$dataset $(TEST_ARGS) | tee "results_$$(basename $$dataset .dat).csv"; \
	done

check_vlen:
	@vlen=$$(cat /sys/devices/virtual/riscv_isa 2>/dev/null | grep -o 'v256\|v512\|v128' || echo "unknown"); \
	echo "Detected VLEN: $$vlen"; \
	# You can add logic here to assert vlen matches expected
