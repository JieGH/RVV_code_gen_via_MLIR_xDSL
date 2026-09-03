#!/bin/bash

mkdir -p $1
bits=$2    #128, 256
gather=$3  #0-bcast, 1-gather, 2-direct
swap=$4    #0-loadAB, 1-loadBA
fp=$5      #fp32, fp16
rep=$6     #number of reps
mode=ALL

if [ ${gather} -eq 2 ] ; then 
	if [ ${swap} -eq 1 ] ; then
	    echo "ERROR: Direct version can no swap A and B loads"
	    exit 1
	fi
fi

if [ ${gather} -eq 0 ]; then
	gg="bcast"
fi
if [ ${gather} -eq 1 ]; then
	gg="gather"
fi
if [ ${gather} -eq 2 ]; then
	gg="direct"
fi

if [ ${swap} -eq 0 ]; then
	ss="loadAB"
else
	ss="loadBA"
fi

mrini=1
nrini=1
step=1

if [ ${bits} -eq 128 ]; then
        if [ ${fp} -eq 32 ]; then
	    mr=32
	    nr=32
	else
	    mr=64
	    nr=64
	fi
else
        if [ ${fp} -eq 32 ]; then
	    mr=64
	    nr=64
	else
	    mr=128
	    nr=128
	fi
fi




ff=$1/${bits}_fp${fp}_${gg}_${ss}_${rep}.dat
make clean MR=${mr} NR=${nr} BITS=${bits} MODE=${mode} GATHER=${gather} SWAP=${swap}
make MR=${mr} NR=${nr} BITS=${bits} MODE=${mode} GATHER=${gather} SWAP=${swap}
./test_uk ${mrini} ${mr} ${nrini} ${nr} ${rep} 0 1000 > ${ff}
