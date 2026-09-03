#/bin/bash

ini=8
end=48
step=8
ininr=4
endnr=48
stepnr=4
mkdir -p out_models
for model in resnet50 square googlenet vgg16;
do
for mode in BASE LDX OPT UNROLL;
do
    for swap in 0 1;
      do
      if [ ${swap} -eq 1 ]; then
            mm="loadBA"
          else
            mm="loadAB"
          fi
        for gather in 0 1 2;
        do
	  if [ ${gather} -eq 1 ]; then
            gg="gather"
          else
	    if [ ${gather} -eq 2 ]; then
              gg="macc"
            else
              gg="bcast"
	    fi
          fi
	  for r in 512;
          do
	    #echo "" > ${ff}
            for mr in $(seq ${ini} ${step} ${end});
            do
               for nr in $(seq ${ininr} ${stepnr} ${endnr});
               do 
	               ff=out_models/${model}_${mode}_${mr}x${nr}_${mm}_${gg}_${r}.dat
		       make clean
		       make MR=${mr} NR=${nr} OPT=${mode} GATHER=${gather} SWAP=${swap} SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO
		       echo "./run_gemm.sh cnn_models/${model}.dat ${ff}"
		       ./run_gemm.sh cnn_models/${model}.dat ${ff}
	      done
	    done
         done
      done
   done
done
done


