#/bin/bash

ini=8
end=48
step=8
ininr=4
endnr=48
stepnr=4
for model in resnet50; # square googlenet vgg16;
do
for mode in UNROLL; #LDX OPT UNROLL; # BASE;
do
   for mr in $(seq ${ini} ${step} ${end});
    do
      for nr in $(seq ${ininr} ${stepnr} ${endnr});
      do 
        for gather in 0 1;
        do
	  if [ ${gather} -eq 1 ]; then
            gg="gather"
          else
            gg="bcast"
          fi
          for swap in 0 1;
             do
             if [ ${swap} -eq 1 ]; then
               mm="loadBA"
              else
              mm="loadAB"
              fi
	      for r in 512;
              do
	      ff=out_models/${model}_${mode}_${mr}x${nr}_${mm}_${gg}_${r}.dat
	      #	       touch out_models/${model}_${mode}_${mm}_${gg}.dat
	      if test -f ${ff}; then
   	         echo "${model} ${mode} ${mm} ${gg} ${mr} ${nr}"
	         cat ${ff} | cut -d";" -f5 > tmp_${mr}_${nr}_${gg}_${mm}.dat
	      fi
	      done #r
	    done # swap
         done # gather
	 if test -f tmp_${mr}_${nr}_bcast_loadAB.dat; then
	     paste tmp_${mr}_${nr}_bcast_loadAB.dat tmp_${mr}_${nr}_bcast_loadBA.dat tmp_${mr}_${nr}_gather_loadAB.dat tmp_${mr}_${nr}_gather_loadBA.dat > out_models/${model}_${mode}_${mr}_${nr}.dat
	     paste tmp_${mr}_${nr}_bcast_loadAB.dat tmp_${mr}_${nr}_bcast_loadBA.dat tmp_${mr}_${nr}_gather_loadAB.dat tmp_${mr}_${nr}_gather_loadBA.dat
	     rm tmp_${mr}_${nr}_bcast_loadAB.dat tmp_${mr}_${nr}_bcast_loadBA.dat tmp_${mr}_${nr}_gather_loadAB.dat tmp_${mr}_${nr}_gather_loadBA.dat
         fi
	 read 
     done # nr
   done #nr
done #mode
done #model


