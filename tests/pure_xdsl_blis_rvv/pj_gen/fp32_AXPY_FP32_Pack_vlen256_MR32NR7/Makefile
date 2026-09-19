CXX = g++
CC = gcc

# Issue #14: instrumentation switch.
#   INSTRUMENTED=1 (default) keeps the per-tile chrono pairs and the perf_event_open
#     reads. That is the build that produces the pack/compute breakdown and the
#     counter log, and it pays ~125-160 ns of timer tax per microkernel tile.
#   INSTRUMENTED=0 compiles them out. Only the outer wall clock survives, so the
#     breakdown columns are nan and the logs are written under a _noinstr name, but
#     RVV_Time_ms then carries no timer tax and the timed region has the same shape
#     as the reference design's.
# Objects and binary are kept apart per mode so the two can never be mixed by a
# stale .o, and so both binaries can sit side by side for an A/B run.
INSTRUMENTED ?= 1
BUILD_DIR = build_instr$(INSTRUMENTED)

# Basic flags for RISC-V Vector extension
CFLAGS = -O3 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing -DRVV_INSTRUMENTED=$(INSTRUMENTED)
CXXFLAGS = -O2 -std=c++17 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing -fno-var-tracking-assignments --param ggc-min-expand=1 --param ggc-min-heapsize=32768 -DRVV_INSTRUMENTED=$(INSTRUMENTED)

# All generated files are C++ now, plus model_level.c
CXX_SRCS = $(wildcard *.cpp)
CXX_OBJS = $(addprefix $(BUILD_DIR)/,$(CXX_SRCS:.cpp=.o))

C_SRCS = ../../modelLevel/model_level.c
C_OBJS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SRCS:.c=.o)))

TARGET = test_gemm_fp32_AXPY_FP32_Pack.x
RESULT_SUFFIX =
ifneq ($(INSTRUMENTED),1)
TARGET := $(TARGET:.x=_noinstr.x)
RESULT_SUFFIX = _noinstr
endif
# Issue #27: a --ref-protocol run gets its own results file too, like its logs.
ifneq (,$(findstring --ref-protocol,$(TEST_ARGS)))
RESULT_SUFFIX := $(RESULT_SUFFIX)_refproto
endif

all: $(TARGET)

$(TARGET): $(CXX_OBJS) $(C_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: ../../modelLevel/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build_instr* *.o test_gemm*.x

DATASETS_PATTERN ?= ../../datasets/issue2_square_bert_32x7.dat

run: $(TARGET)
	@mkdir -p logs
	@for dataset in $(DATASETS_PATTERN); do \
		echo "Running dataset: $$dataset"; \
		./$(TARGET) $$dataset $(TEST_ARGS) | tee "results_$$(basename $$dataset .dat)$(RESULT_SUFFIX).csv"; \
	done

check_vlen:
	@vlen=$$(cat /sys/devices/virtual/riscv_isa 2>/dev/null | grep -o 'v256\|v512\|v128' || echo "unknown"); \
	echo "Detected VLEN: $$vlen"; \
	# You can add logic here to assert vlen matches expected
