DTYPE = -DFP32
DT=fp32
#MR = 8
#NR = 12

#BLIS_HOME    = ${BLISHOME}
BLIS_HOME    = /home/adcastel/opt/blis
INCLUDE_BLIS = -I$(BLIS_HOME)/include/blis/
LIB_BLIS     = -lblis -L$(BLIS_HOME)/lib

INCLUDE = ${INCLUDE_BLIS}
LIBS    = ${LIB_BLIS} -lm

#KPATH=src/RVV_256_UNROLL/${DT}/${MR}x${NR}/
KPATH=../kernels/RVV_256_${OPT}/${DT}/${MR}x${NR}/
FILE=kernels_RVV_${MR}x${NR}_${DT}

ifeq ($(SWAP),1)
    ifeq ($(GATHER),1)
	ESP=/loadBA/gather/
	B=gather
	DB=-Dgather
    else
	ESP=/loadBA/bcast/
	B=bcast
	DB=-Dbcast
    endif
    L=loadBA
    DL=-DloadBA
else
   ifeq ($(GATHER),1)
        ESP=/loadAB/gather/
        B=gather
        DB=-Dgather
    else
        ifeq ($(GATHER),2)
            ESP=/loadAB/macc/
            B=macc
            DB=-Dmacc
        else
            ESP=/loadAB/bcast/
            B=bcast
            DB=-Dbcast
	endif
    endif
    L=loadAB
    DL=-DloadAB
endif


#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
CC       =  gcc
CLINKER  =  gcc
#------------------------------------------

OBJDIR = build
BIN    = test_gemm.x
_OBJ2   = gemm_blis.o inutils.o sutils.o test_gemm.o model_level.o 

#------------------------------------------
#| PATHS CONFIGURE                        |
#------------------------------------------
vpath %.c ./src
vpath %.h ./src

vpath %.c ./modelLevel
vpath %.h ./modelLevel

#vpath %.c ./src/ARMv8
#vpath %.h ./src/ARMv8
#vpath %.c ./src/RVV_256/fp32/${MR}x${NR}/${L}/${B}
#vpath %.h ./src/RVV_256/fp32/${MR}x${NR}/${L}/${B}
#vpath %.c ./src/RVV_256
#vpath %.h ./src/RVV_256
kernels=${KPATH}/${ESP}/${FILE}.o
vpath %.c .${KPATH}/${ESP}
vpath %.h .${KPATH}/${ESP}
#------------------------------------------

#------------------------------------------
#| COMPILER FLAGS                         |
#------------------------------------------

ifeq ($(RUN_MODE), FAMILY)
  MODE=-DFAMILY
else ifeq ($(RUN_MODE), FAMILY_BLIS)
  MODE=-DFAMILY_BLIS
else ifeq ($(RUN_MODE), FAMILY_EXO)
  MODE=-DFAMILY_EXO
else ifeq ($(RUN_MODE), BLIS)
  MODE=-DBLIS
endif

#Dependes Arquitecture Mode
ifeq ($(SIMD_MODE), ARMv8)
  SIMD  = -DARMv8
  FLAGS = -O3 -march=armv8-a+simd+fp -fopenmp
  _OBJ += gemm_blis_neon_fp32.o 
else ifeq ($(SIMD_MODE), ARMv8_EXO)
  SIMD  = -DARMv8_EXO
  FLAGS = -O3 -march=armv8-a+simd+fp
  _OBJ += exo_matrix.o kernel_col.o

else ifeq ($(SIMD_MODE), RVV_EXO)
  SIMD  = -DRVV_EXO
  FLAGS = -O3 -march=rv64gcvzfh -mabi=lp64d -Wfatal-errors 
  _OBJ = exo_matrix_RVV_${DT}.o ${FILE}.o 
  _OBJ += ${_OBJ2}
else
  FLAGS = -O3 
endif

#ifeq ($(RUN_MODE), FAMILY_BLIS)
#  # FIX for all BLIS MR,NR Values!
#  ifeq ($(SIMD_MODE), ARMv8)
#    MR=8
#    NR=12
#  else ifeq ($(SIMD_MODE), ARMv8_EXO)
#    MR=12
#    NR=8
#  else
#  endif	
#endif

OPTFLAGS = $(FLAGS) -DCHECK -DMR=$(MR) -DNR=$(NR) -DKR=1 $(DL) $(DB) $(MODE) $(SIMD) $(DTYPE) -D${OPT}
#------------------------------------------

OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))
default: $(OBJDIR)/$(BIN)

$(OBJDIR)/%.o:%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS) -I$(PWD)/$(KPATH)/${ESP}

        #$(kernels):
build/${FILE}.o: 
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $(KPATH)/${ESP}/${FILE}.c -o build/${FILE}.o


build/exo_matrix_RVV_${DT}.o:
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c ${KPATH}/${ESP}/exo_matrix_RVV_${DT}.c -o build/exo_matrix_RVV_${DT}.o

$(OBJDIR)/$(BIN): $(OBJ) 
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS)


clean:
	rm $(OBJDIR)/* 

