CC=gcc 

#CFLAGS= -O3 -march=rv64gc0p7v
CFLAGS= -O3 -march=rv64gcv -mabi=lp64d -Wfatal-errors
DT=fp32

ifeq ($(BITS), 256)
	B=256
else
	B=128
endif

KPATH=../kernels/RVV_${B}_${MODE}/${DT}/${MR}x${NR}/

ifeq ($(MODE), ALL)
    FILE=kernels_RVV_${DT}
else
    FILE=kernels_RVV_${MR}x${NR}_${DT}
endif

ifeq ($(SWAP), 1)
        ifeq ($(GATHER), 1)
	    ESP=/loadBA/gather/

            BCAS=-DGATHER
        else
	    ESP=/loadBA/bcast/
            BCAS=-DBCAST
        endif

        SWAP=-DSWAPAB

else
        ifeq ($(GATHER), 1)
            BCAS=-DGATHER
	    ESP=/loadAB/gather/
        else
            ifeq ($(GATHER), 0)
	        ESP=/loadAB/bcast/
                BCAS=-DBCAST
	    else
	        ESP=/loadAB/macc/
                BCAS=-DMACC
	    endif
        endif
        SWAP=
endif

kernels=${KPATH}/${ESP}/${FILE}.o
vpath %.c ./${KPATH}/${ESP}
vpath %.h ./${KPATH}/${ESP}
OBJECTS := ${KPATH}/${ESP}/exo_matrix_RVV_${DT}.o $(kernels)

all:main

main: $(OBJECTS)
	$(CC) $(CFLAGS) main.c -o test_uk $(OBJECTS) -I$(PWD) -I$(PWD)/$(KPATH)/${ESP} -L$(PWD) -D$(MODE)

$(kernels):
	$(CC) $(CFLAGS) -o $(kernels) -c $*.c

$(KPATH)/$(ESP)/exo_matrix_RVV_${DT}.o:
	$(CC) $(CFLAGS) -o ${KPATH}/${ESP}/exo_matrix_RVV_${DT}.o -c $*.c 
	#$(CC) $(CFLAGS) -c ${KPATH}/${ESP}/exo_matrix_RVV_${DT}.c 

.c.o:
	$(CC) $(CFLAGS) -c $*.c $(SWAP) $(BCAS) -I$(PWD)/$(KPATH) -L$(PWD)/$(KPATH)

clean:
	rm *.o test_uk ${OBJECTS}
