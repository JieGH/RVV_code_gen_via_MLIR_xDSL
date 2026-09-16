CC = gcc
CXX = g++
CFLAGS = -O3 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing -DFP32 -DFAMILY_EXO -DRVV_EXO -DMR=16 -DNR=15 -I./ -I./src -I./modelLevel
CXXFLAGS = -O3 -std=c++17 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing -DFP32 -DFAMILY_EXO -DRVV_EXO -DMR=16 -DNR=15 -I./ -I./src -I./modelLevel

TARGET = test_gemm

OBJS = src/test_gemm.o src/gemm_blis.o src/inutils.o src/sutils.o \
       modelLevel/model_level.o \
       kernels_RVV_16x16_fp32.o exo_matrix_RVV_fp32.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)
