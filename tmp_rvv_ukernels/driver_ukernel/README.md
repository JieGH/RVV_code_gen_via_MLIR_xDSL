# Driver Ukernel

A benchmark for micro-kernel in a solo-mode

## How to use it

``make MR=mr NR=nr MODE=mode GATHER=gather SWAP=swap``

- mr is the first micro-kernel dimension
- nr is the second micro-kernel dimension
- mode [LDX|OPT|BASE] are the three generated micro-kernels
- gather [0|1|2] is the mode of loading the B elements. 0 is for broadcast, 1 for gather, and 2 for direct
- swap [0|1] indicates if the loads of A and B are swapped. 0 loads first A, and 1 loads first B

``./test_uk mri mre nri nre k beta reps``

- mri and mre are the init value and end value of mr
- nri and nre are the init value and end value of nr
- k is the iterations of the k-loop
- beta [0|1] is the value of beta
- reps is the number of repetitions

You can compile and execute all the variants with the ``execute_all.sh`` script.

Note that the kernels are in the upper directory.
