# debug perfromance differnces

Status: Done Unsolved
Due: July 23, 2026 → July 28, 2026

The debug why perf of pure xdsl is slower than priv ref

Reason guessing:

profiling method is difference

compare kernel head to kead run code to verifyu 

The major changes of the new design is the performance iteration changed from 5 to 500, rge 5 case is two low, 

```jsx
▶ B2_M5 (384x384x64)     | MR16_NR15  11.49 Tot GOPs |  12.83 Ker GOPs |   2.2 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7 (64x384x384)     | MR16_NR15   4.58 Tot GOPs |   7.24 Ker GOPs |   0.5 GB/s RVV Pk |   1.0 GB/s Scl Pk | ⏭️ SKIP
Connection to jbpi2 closed.
```

Priviouse results 

| **B2 (B2_M5)** | 384×384×64 | ~12.5 | 16x15 | **12.96** | `MR16_NR15` | +3.7% |
| --- | --- | --- | --- | --- | --- | --- |
| **B3 (B3_M7)** | 64×384×384 | ~8.8 | 16x15 | **10.86** | `MR16_NR11` | +23.4% |

You can see, if i force kernel to 16x15, the performance is very close. 

B2 pro_xdsl is 12.5, pure_blis is 12.83, 

B3 pri is 8.8m pure is 7.24

if i **force a 16x11**

500 iterations 

▶ B3_M7 (64x384x384)     | MR16_NR11   4.02 Tot GOPs |   6.38 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP

▶ B3_M7 (64x384x384)     | MR16_NR11   4.22 Tot GOPs |   6.68 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP

1000 iterations

▶ B3_M7 (64x384x384)     | MR16_NR11   4.11 Tot GOPs |   6.53 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP

force a **16x15, close to 12.5 and 8,8 at ref design** 

▶ B2_M5 (384x384x64)     | MR16_NR15  11.64 Tot GOPs |  12.87 Ker GOPs |   2.6 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7 (64x384x384)     | MR16_NR15   4.71 Tot GOPs |   7.55 Ker GOPs |   0.5 GB/s RVV Pk |   1.0 GB/s Scl Pk | ⏭️ SKIP

> When run independednly force a **16x15,** ,  b3 has cloer  perf as privious ref design, 

Run the pure full design

July 26, 2026 9:08 PM (GMT+8) Early results shows 

B1_M123,1024,384,1024,MR32_NR7,SKIP,6.98863,8.47181,0.32079,0.274934,115.231,0

the kernel 8.5GFLOP, pri 8

For B3 , pri 16x15 is ~8.8, obtain best 16x14 of 7,4

- Early log
    
    16x15
    
    B3_M7,64,384,384,MR16_NR15,SKIP,4.44503,7.1895,0.449805,0.482407,4.24617,0
    
    16x11
    
    B3_M7,64,384,384,MR16_NR11,SKIP,3.9686,6.27278,0.418815,0.504877,4.75592,0
    
    16x7
    
    B3_M7,64,384,384,MR16_NR7,SKIP,5.29353,9.23182,0.49027,0.564591,3.56556,0
    
    16x6
    
    B3_M7,64,384,384,MR16_NR6,SKIP,6.29989,10.1509,0.6801,0.603957,2.99599,0
    
    - 
        
        ```jsx
        [jlei@fej logs]$ cat raw_fp32_AXPY_FP32_Pack.txt
        B3_M7,64,384,384,MR16_NR16,SKIP,3.14824,4.44578,0.414117,0.580298,5.9952,0
        B3_M7,64,384,384,MR16_NR15,SKIP,4.44503,7.1895,0.449805,0.482407,4.24617,0
        B3_M7,64,384,384,MR16_NR14,SKIP,4.66875,7.43097,0.488768,0.451532,4.0427,0
        B3_M7,64,384,384,MR16_NR13,SKIP,4.56477,7.3097,0.472839,0.382552,4.13479,0
        B3_M7,64,384,384,MR16_NR12,SKIP,4.37527,7.0019,0.453073,0.377782,4.31387,0
        B3_M7,64,384,384,MR16_NR11,SKIP,3.9686,6.27278,0.418815,0.504877,4.75592,0
        B3_M7,64,384,384,MR16_NR10,SKIP,3.76153,5.99594,0.391068,0.407938,5.01773,0
        B3_M7,64,384,384,MR16_NR9,SKIP,3.7998,6.24908,0.375872,0.489919,4.9672,0
        B3_M7,64,384,384,MR16_NR8,SKIP,3.85209,6.1803,0.396991,0.938734,4.89977,0
        B3_M7,64,384,384,MR16_NR7,SKIP,5.29353,9.23182,0.49027,0.564591,3.56556,0
        B3_M7,64,384,384,MR16_NR6,SKIP,6.29989,10.1509,0.6801,0.603957,2.99599,0
        B3_M7,64,384,384,MR16_NR5,SKIP,5.38767,8.61836,0.587101,0.57634,3.50326,0
        B3_M7,64,384,384,MR16_NR4,SKIP,4.92727,7.60463,0.578746,0.537012,3.83059,0
        B3_M7,64,384,384,MR16_NR3,SKIP,4.66386,6.73289,0.655105,0.588331,4.04694,0
        B3_M7,64,384,384,MR16_NR2,SKIP,3.68217,5.3432,0.515681,0.439435,5.12587,0
        B3_M7,64,384,384,MR16_NR1,SKIP,2.30602,3.12843,0.404352,0.446434,8.18481,0
        B3_M7,64,384,384,MR15_NR16,SKIP,2.36844,3.04746,0.409932,0.38031,7.9691,0
        B3_M7,64,384,384,MR15_NR15,SKIP,3.05058,4.09718,0.464884,0.390225,6.18715,0
        B3_M7,64,384,384,MR15_NR14,SKIP,3.06349,4.17189,0.448873,0.365947,6.16106,0
        B3_M7,64,384,384,MR15_NR13,SKIP,3.00388,4.08026,0.443955,0.37163,6.28334,0
        B3_M7,64,384,384,MR15_NR12,SKIP,3.02563,4.07048,0.46145,0.372229,6.23816,0
        B3_M7,64,384,384,MR15_NR11,SKIP,2.90964,3.99504,0.417852,0.40573,6.48684,0
        B3_M7,64,384,384,MR15_NR10,SKIP,2.84727,3.95991,0.395486,0.368688,6.62894,0
        B3_M7,64,384,384,MR15_NR9,SKIP,2.73669,3.86187,0.366052,0.342961,6.89679,0
        B3_M7,64,384,384,MR15_NR8,SKIP,2.79824,3.82724,0.410574,0.598171,6.7451,0
        B3_M7,64,384,384,MR15_NR7,SKIP,3.86377,5.2096,0.610309,0.539365,4.88497,0
        B3_M7,64,384,384,MR15_NR6,SKIP,3.70877,4.98359,0.594529,0.498734,5.08911,0
        B3_M7,64,384,384,MR15_NR5,SKIP,3.39674,4.63937,0.519232,0.530246,5.55661,0
        B3_M7,64,384,384,MR15_NR4,SKIP,3.13631,4.19672,0.514575,0.535753,6.01803,0
        B3_M7,64,384,384,MR15_NR3,SKIP,3.00388,3.82989,0.608151,0.625981,6.28333,0
        B3_M7,64,384,384,MR15_NR2,SKIP,2.44579,3.15169,0.481242,0.499653,7.71707,0
        B3_M7,64,384,384,MR15_NR1,SKIP,1.48951,1.79148,0.427198,0.476875,12.6715,0
        B3_M7,64,384,384,MR14_NR16,SKIP,2.33943,3.1108,0.361675,0.453473,8.06795,0
        B3_M7,64,384,384,MR14_NR15,SKIP,3.1624,4.26134,0.478614,0.423176,5.96836,0
        B3_M7,64,384,384,MR14_NR14,SKIP,3.04746,4.14322,0.448857,0.368164,6.19348,0
        B3_M7,64,384,384,MR14_NR13,SKIP,2.88395,4.02339,0.394331,0.369802,6.54463,0
        B3_M7,64,384,384,MR14_NR12,SKIP,2.9804,4.08668,0.429099,0.376202,6.33282,0
        B3_M7,64,384,384,MR14_NR11,SKIP,2.82865,3.95563,0.385564,0.418539,6.67258,0
        B3_M7,64,384,384,MR14_NR10,SKIP,2.73181,3.87451,0.359452,0.539001,6.9091,0
        B3_M7,64,384,384,MR14_NR9,SKIP,2.831,4.02904,0.371417,0.363761,6.66703,0
        B3_M7,64,384,384,MR14_NR8,SKIP,2.73307,3.83442,0.372806,0.546719,6.90593,0
        B3_M7,64,384,384,MR14_NR7,SKIP,3.99245,5.27686,0.677374,0.519867,4.72751,0
        B3_M7,64,384,384,MR14_NR6,SKIP,3.77709,4.99279,0.641663,0.568013,4.99706,0
        B3_M7,64,384,384,MR14_NR5,SKIP,3.47749,4.71357,0.545471,0.563372,5.42758,0
        B3_M7,64,384,384,MR14_NR4,SKIP,3.27069,4.31048,0.569002,0.513523,5.77076,0
        B3_M7,64,384,384,MR14_NR3,SKIP,3.08296,3.921,0.634054,0.633854,6.12215,0
        B3_M7,64,384,384,MR14_NR2,SKIP,2.5744,3.26813,0.545911,0.527556,7.33157,0
        B3_M7,64,384,384,MR14_NR1,SKIP,1.54827,1.86852,0.436526,0.503234,12.1906,0
        B3_M7,64,384,384,MR13_NR16,SKIP,2.24117,2.85905,0.399419,0.751781,8.42165,0
        B3_M7,64,384,384,MR13_NR15,SKIP,2.89362,3.8364,0.457971,0.533349,6.52275,0
        B3_M7,64,384,384,MR13_NR14,SKIP,2.78426,3.75104,0.418713,0.474434,6.77895,0
        B3_M7,64,384,384,MR13_NR13,SKIP,2.81861,3.81442,0.419127,0.397639,6.69635,0
        B3_M7,64,384,384,MR13_NR12,SKIP,2.8986,3.86749,0.452078,0.400122,6.51154,0
        B3_M7,64,384,384,MR13_NR11,SKIP,2.71641,3.74459,0.38369,0.37334,6.94827,0
        B3_M7,64,384,384,MR13_NR10,SKIP,2.62168,3.6317,0.365883,0.398506,7.19935,0
        [jlei@fej logs]$
        ```
        

When force pure design for B3_M7 to 16x15, perf is close, hence same kernel perf similar 

xdsl privious ref design is 8..8,  i obtained 

The pure design gives closer to privious design, i can assum the kernel performe the same with same kernel dimention 

```jsx
▶ B3_M7_1 (64x384x384)   | MR16_NR15   5.05 Tot GOPs |   8.57 Ker GOPs |   0.5 GB/s RVV Pk |   0.8 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_2 (64x384x384)   | MR16_NR15   4.95 Tot GOPs |   8.10 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_3 (64x384x384)   | MR16_NR15   5.38 Tot GOPs |   9.11 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_4 (64x384x384)   | MR16_NR15   4.78 Tot GOPs |   7.90 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_5 (64x384x384)   | MR16_NR15   5.31 Tot GOPs |   8.98 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_6 (64x384x384)   | MR16_NR15   5.06 Tot GOPs |   8.25 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_7 (64x384x384)   | MR16_NR15   5.05 Tot GOPs |   8.65 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_8 (64x384x384)   | MR16_NR15   4.82 Tot GOPs |   7.94 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_9 (64x384x384)   | MR16_NR15   5.20 Tot GOPs |   8.80 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
▶ B3_M7_10 (64x384x384)  | MR16_NR15   4.78 Tot GOPs |   7.86 Ker GOPs |   0.5 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
```

But search can find out better kernel give better perf 

We force pure design run 16x15 kernel for B2 and B3

| **B2 (B2_M5)** | 384×384×64 | ~12.5 | 16x15 |
| --- | --- | --- | --- |
| **B3 (B3_M7)** | 64×384×384 | ~8.8 | 16x15 |

July 26, 2026 9:44 PM (GMT+8) add a reset c fix 

- repeat B3 and B2 with forced 16x15, same dimention w/ ref design
    
    
    | **Benchmark** | **Machine** | **Run 1** | **Run 2** | **Combined Mean** |
    | --- | --- | --- | --- | --- |
    | B3_M7 (64×384×384) | jbpi2 | 8.69 | 8.75 | **8.72** |
    | B3_M7 (64×384×384) | jbpi | 8.22 | 8.75 | **8.48** |
    | B2_M5 (384×384×64) | jbpi2 | 12.50 | 12.69 | **12.59** |
    | B2_M5 (384×384×64) | jbpi | 12.59 | 12.69 | **12.64** |
    
    ```jsx
    jbpi2
    ▶ B3_M7_1 (64x384x384)   | MR16_NR15   6.94 Tot GOPs |  10.63 Ker GOPs |   0.8 GB/s RVV Pk |   0.9 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR15   7.39 Tot GOPs |  11.08 Ker GOPs |   0.9 GB/s RVV Pk |   0.9 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR15   4.96 Tot GOPs |   8.50 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR15   5.05 Tot GOPs |   8.19 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR15   5.12 Tot GOPs |   8.67 Ker GOPs |   0.5 GB/s RVV Pk |   0.8 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR15   4.68 Tot GOPs |   7.69 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR15   4.87 Tot GOPs |   8.37 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR15   5.00 Tot GOPs |   8.05 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR15   4.89 Tot GOPs |   8.27 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR15   4.59 Tot GOPs |   7.41 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_1 (384x384x64)   | MR16_NR15  11.53 Tot GOPs |  12.87 Ker GOPs |   2.1 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_2 (384x384x64)   | MR16_NR15  11.56 Tot GOPs |  12.88 Ker GOPs |   2.2 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_3 (384x384x64)   | MR16_NR15  11.60 Tot GOPs |  12.92 Ker GOPs |   2.3 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_4 (384x384x64)   | MR16_NR15  11.08 Tot GOPs |  12.47 Ker GOPs |   1.8 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_5 (384x384x64)   | MR16_NR15  11.15 Tot GOPs |  12.55 Ker GOPs |   1.8 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_6 (384x384x64)   | MR16_NR15  11.39 Tot GOPs |  12.75 Ker GOPs |   2.0 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_7 (384x384x64)   | MR16_NR15  10.83 Tot GOPs |  12.19 Ker GOPs |   1.7 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_8 (384x384x64)   | MR16_NR15  10.52 Tot GOPs |  11.85 Ker GOPs |   1.6 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_9 (384x384x64)   | MR16_NR15  10.23 Tot GOPs |  11.58 Ker GOPs |   1.4 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_10 (384x384x64)  | MR16_NR15  11.54 Tot GOPs |  12.91 Ker GOPs |   2.1 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    jbpi2
    ▶ B3_M7_1 (64x384x384)   | MR16_NR15   5.14 Tot GOPs |   8.49 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR15   4.78 Tot GOPs |   7.76 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR15   5.04 Tot GOPs |   8.61 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR15   4.89 Tot GOPs |   7.95 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR15   5.14 Tot GOPs |   8.77 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR15   5.16 Tot GOPs |   8.31 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR15   8.59 Tot GOPs |  12.58 Ker GOPs |   1.1 GB/s RVV Pk |   0.9 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR15   4.94 Tot GOPs |   7.95 Ker GOPs |   0.5 GB/s RVV Pk |   0.9 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR15   5.45 Tot GOPs |   9.20 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR15   4.81 Tot GOPs |   7.83 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_1 (384x384x64)   | MR16_NR15  10.91 Tot GOPs |  12.21 Ker GOPs |   1.8 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_2 (384x384x64)   | MR16_NR15  11.38 Tot GOPs |  12.68 Ker GOPs |   2.1 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_3 (384x384x64)   | MR16_NR15  11.56 Tot GOPs |  12.86 Ker GOPs |   2.3 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_4 (384x384x64)   | MR16_NR15  11.54 Tot GOPs |  12.83 Ker GOPs |   2.3 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_5 (384x384x64)   | MR16_NR15  11.15 Tot GOPs |  12.48 Ker GOPs |   1.9 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_6 (384x384x64)   | MR16_NR15  11.55 Tot GOPs |  12.86 Ker GOPs |   2.2 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_7 (384x384x64)   | MR16_NR15  11.38 Tot GOPs |  12.69 Ker GOPs |   2.1 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_8 (384x384x64)   | MR16_NR15  11.55 Tot GOPs |  12.85 Ker GOPs |   2.2 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_9 (384x384x64)   | MR16_NR15  11.52 Tot GOPs |  12.84 Ker GOPs |   2.2 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_10 (384x384x64)  | MR16_NR15  11.29 Tot GOPs |  12.60 Ker GOPs |   2.1 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi2 closed.
    
    jbpi
    ▶ B3_M7_1 (64x384x384)   | MR16_NR15   4.93 Tot GOPs |   8.25 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR15   4.97 Tot GOPs |   8.16 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR15   5.05 Tot GOPs |   8.58 Ker GOPs |   0.5 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR15   4.89 Tot GOPs |   8.03 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR15   4.99 Tot GOPs |   8.48 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR15   4.78 Tot GOPs |   7.88 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR15   5.17 Tot GOPs |   8.75 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR15   4.65 Tot GOPs |   7.69 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR15   5.00 Tot GOPs |   8.54 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR15   4.79 Tot GOPs |   7.85 Ker GOPs |   0.5 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_1 (384x384x64)   | MR16_NR15  11.53 Tot GOPs |  12.92 Ker GOPs |   2.0 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_2 (384x384x64)   | MR16_NR15  11.60 Tot GOPs |  13.00 Ker GOPs |   2.0 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_3 (384x384x64)   | MR16_NR15  11.45 Tot GOPs |  12.88 Ker GOPs |   1.9 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_4 (384x384x64)   | MR16_NR15  11.54 Tot GOPs |  12.97 Ker GOPs |   1.9 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_5 (384x384x64)   | MR16_NR15  11.42 Tot GOPs |  12.90 Ker GOPs |   1.8 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_6 (384x384x64)   | MR16_NR15  11.00 Tot GOPs |  12.43 Ker GOPs |   1.7 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_7 (384x384x64)   | MR16_NR15  11.01 Tot GOPs |  12.41 Ker GOPs |   1.7 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_8 (384x384x64)   | MR16_NR15  10.69 Tot GOPs |  12.11 Ker GOPs |   1.5 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_9 (384x384x64)   | MR16_NR15  10.99 Tot GOPs |  12.35 Ker GOPs |   1.8 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_10 (384x384x64)  | MR16_NR15  10.59 Tot GOPs |  11.92 Ker GOPs |   1.6 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi closed.
    jbpi
    ▶ B3_M7_1 (64x384x384)   | MR16_NR15   4.89 Tot GOPs |   8.32 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR15   4.56 Tot GOPs |   7.55 Ker GOPs |   0.4 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR15   4.92 Tot GOPs |   8.43 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR15   4.71 Tot GOPs |   7.70 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR15   5.16 Tot GOPs |   8.72 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR15   4.59 Tot GOPs |   7.50 Ker GOPs |   0.5 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR15   8.17 Tot GOPs |  12.22 Ker GOPs |   1.0 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR15   6.61 Tot GOPs |  10.28 Ker GOPs |   0.7 GB/s RVV Pk |   0.9 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR15   5.40 Tot GOPs |   9.03 Ker GOPs |   0.5 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR15   4.73 Tot GOPs |   7.71 Ker GOPs |   0.5 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_1 (384x384x64)   | MR16_NR15  11.00 Tot GOPs |  12.39 Ker GOPs |   1.7 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_2 (384x384x64)   | MR16_NR15  11.55 Tot GOPs |  12.88 Ker GOPs |   2.2 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_3 (384x384x64)   | MR16_NR15  11.54 Tot GOPs |  12.87 Ker GOPs |   2.2 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_4 (384x384x64)   | MR16_NR15  11.48 Tot GOPs |  12.85 Ker GOPs |   2.0 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_5 (384x384x64)   | MR16_NR15  11.48 Tot GOPs |  12.82 Ker GOPs |   2.1 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_6 (384x384x64)   | MR16_NR15  11.32 Tot GOPs |  12.61 Ker GOPs |   2.1 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_7 (384x384x64)   | MR16_NR15  11.59 Tot GOPs |  12.84 Ker GOPs |   2.4 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_8 (384x384x64)   | MR16_NR15  10.97 Tot GOPs |  12.21 Ker GOPs |   2.0 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_9 (384x384x64)   | MR16_NR15  11.34 Tot GOPs |  12.60 Ker GOPs |   2.3 GB/s RVV Pk |   1.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B2_M5_10 (384x384x64)  | MR16_NR15  11.55 Tot GOPs |  12.82 Ker GOPs |   2.4 GB/s RVV Pk |   1.3 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi closed.
    ```
    

- Repeat B2 with forced 16x11, my results at [Cache config search and FP32 AXBY BLIS](https://app.notion.com/p/Cache-config-search-and-FP32-AXBY-BLIS-39868a923c2a8007b2f9e4e4a9b1e661?pvs=21)  is 10.8G, but now i obtained aorund 7G, hence priv results is strange, it was 10.8
    
    There is no results from pri_xdsl, hence here prove the 10g pbtained from last note is not funded 
    
    | **Machine** | **Kernel** | **Run 1** | **Run 2** | **Combined** |
    | --- | --- | --- | --- | --- |
    | jbpi2 | MR16_NR15 | 8.69 | 8.75 | **8.72** |
    | jbpi2 | MR16_NR11 | 6.72 | 6.72 | **6.72** |
    | jbpi | MR16_NR15 | 8.22 | 8.75 | **8.48** |
    | jbpi | MR16_NR11 | 7.13 | 6.59 | **6.86** |
    
    ```jsx
    ▶ B3_M7_1 (64x384x384)   | MR16_NR11   4.18 Tot GOPs |   6.74 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR11   4.06 Tot GOPs |   6.59 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR11   4.23 Tot GOPs |   6.97 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR11   4.12 Tot GOPs |   6.69 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR11   4.17 Tot GOPs |   6.86 Ker GOPs |   0.4 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR11   4.10 Tot GOPs |   6.64 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR11   4.08 Tot GOPs |   6.65 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR11   4.08 Tot GOPs |   6.51 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR11   4.32 Tot GOPs |   7.02 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR11   4.02 Tot GOPs |   6.48 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi2 closed.
    ▶ B3_M7_1 (64x384x384)   | MR16_NR11   4.23 Tot GOPs |   6.88 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR11   4.23 Tot GOPs |   6.82 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR11   4.14 Tot GOPs |   6.77 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR11   4.01 Tot GOPs |   6.49 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR11   4.05 Tot GOPs |   6.58 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR11   3.98 Tot GOPs |   6.38 Ker GOPs |   0.4 GB/s RVV Pk |   0.8 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR11   4.08 Tot GOPs |   6.64 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR11   4.04 Tot GOPs |   6.48 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR11   4.10 Tot GOPs |   6.68 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR11   3.99 Tot GOPs |   6.43 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi2 closed.
    
    ▶ B3_M7_1 (64x384x384)   | MR16_NR11   4.08 Tot GOPs |   6.67 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR11   3.98 Tot GOPs |   6.38 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR11   4.19 Tot GOPs |   6.92 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR11   4.04 Tot GOPs |   6.50 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR11   4.06 Tot GOPs |   6.67 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR11   4.28 Tot GOPs |   6.87 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR11   6.34 Tot GOPs |   9.73 Ker GOPs |   0.7 GB/s RVV Pk |   0.9 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR11   3.97 Tot GOPs |   6.38 Ker GOPs |   0.4 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR11   4.17 Tot GOPs |   6.88 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR11   3.89 Tot GOPs |   6.25 Ker GOPs |   0.4 GB/s RVV Pk |   0.4 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi closed.
    ▶ B3_M7_1 (64x384x384)   | MR16_NR11   4.12 Tot GOPs |   6.71 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_2 (64x384x384)   | MR16_NR11   3.96 Tot GOPs |   6.41 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_3 (64x384x384)   | MR16_NR11   4.09 Tot GOPs |   6.75 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_4 (64x384x384)   | MR16_NR11   4.01 Tot GOPs |   6.45 Ker GOPs |   0.4 GB/s RVV Pk |   0.7 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_5 (64x384x384)   | MR16_NR11   4.05 Tot GOPs |   6.63 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_6 (64x384x384)   | MR16_NR11   3.90 Tot GOPs |   6.29 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_7 (64x384x384)   | MR16_NR11   4.09 Tot GOPs |   6.70 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_8 (64x384x384)   | MR16_NR11   4.01 Tot GOPs |   6.47 Ker GOPs |   0.4 GB/s RVV Pk |   0.6 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_9 (64x384x384)   | MR16_NR11   4.23 Tot GOPs |   7.02 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    ▶ B3_M7_10 (64x384x384)  | MR16_NR11   4.01 Tot GOPs |   6.46 Ker GOPs |   0.4 GB/s RVV Pk |   0.5 GB/s Scl Pk | ⏭️ SKIP
    Connection to jbpi closed.
    
    ```
    

<aside>
☝

So far the concultion is pure xdsl under same kernel dimention, produce very similar perf, the privious pure xdsl massive outperfromance privious design is not funded,

But the results of perf is not stable

But the sweep schenem need to be changed to 

</aside>

<aside>
⚠️

Myth, the B3 layout results is very unstable, yet it obained the same perf as pervious

| **B2 (B2_M5)** | 384×384×64 | ~12.5 | 16x15 |
| --- | --- | --- | --- |
| **B3 (B3_M7)** | 64×384×384 | ~8.8 | 16x15 |

```jsx
   Metric Explanations:
   - p5 / p95 : 5th and 95th percentiles (ignores top 5% lucky spikes and bottom 5% OS interrupts)
   - iqr      : Interquartile Range (P75 - P25), shows how tight the middle 50% of runs are clustered
   - s        : Standard deviation of Kernel GOPs across all runs
   - CV%      : Coefficient of Variation (s / median). >15% indicates severe performance instability
   - mx/md    : Max / Median ratio. >1.25 indicates extreme cache sensitivity (thrashing)
   
   Layer Name (M x N x K)       | Unroll | Median GOPs [min, p5, p95, max, iqr, s, CV%, mx/md] | RVV Pack | Scl Pack | IPC   | Status
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
▶ L1 (1x1x1)             | 16x15   0.01 KerGOPs [min:0.0, p5:0.0, p95:0.0, max:0.0, iqr:0.0, s:0.0, CV:7%, mx/md:1.14] |  0.0 GB/s Pk |  0.0 GB/s Scl | ⏭️ SKIP
▶ L2 (3x5x7)             | 16x15   0.39 KerGOPs [min:0.3, p5:0.4, p95:0.4, max:0.4, iqr:0.0, s:0.0, CV:4%, mx/md:1.08] |  0.6 GB/s Pk |  0.6 GB/s Scl | ⏭️ SKIP
▶ L3 (8x8x8)             | 16x15   1.54 KerGOPs [min:0.1, p5:1.4, p95:1.5, max:1.5, iqr:0.0, s:0.1, CV:4%, mx/md:1.00] |  1.2 GB/s Pk |  1.0 GB/s Scl | ⏭️ SKIP
▶ L4 (256x8x32)          | 16x15   4.80 KerGOPs [min:2.7, p5:4.7, p95:4.9, max:4.9, iqr:0.1, s:0.1, CV:3%, mx/md:1.02] |  2.8 GB/s Pk |  1.3 GB/s Scl | ⏭️ SKIP
▶ L5 (8x256x32)          | 16x15   4.63 KerGOPs [min:2.9, p5:4.6, p95:4.6, max:4.7, iqr:0.0, s:0.1, CV:3%, mx/md:1.00] |  2.3 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ L6 (64x64x300)         | 16x15  11.13 KerGOPs [min:9.7, p5:10.7, p95:11.2, max:11.2, iqr:0.0, s:0.1, CV:1%, mx/md:1.01] |  2.1 GB/s Pk |  1.3 GB/s Scl | ⏭️ SKIP
▶ L7 (128x128x128)       | 16x15  13.16 KerGOPs [min:11.1, p5:12.7, p95:13.2, max:13.2, iqr:0.0, s:0.2, CV:1%, mx/md:1.00] |  2.7 GB/s Pk |  1.5 GB/s Scl | ⏭️ SKIP
▶ B3_M7_1 (64x384x384)   | 16x15   6.82 KerGOPs [min:6.2, p5:6.6, p95:7.4, max:12.7, iqr:0.2, s:0.8, CV:11%, mx/md:1.86] |  0.5 GB/s Pk |  0.6 GB/s Scl | ⏭️ SKIP
▶ B3_M7_2 (64x384x384)   | 16x15   7.22 KerGOPs [min:6.4, p5:6.9, p95:12.6, max:12.8, iqr:0.4, s:1.8, CV:24%, mx/md:1.77] |  0.5 GB/s Pk |  0.6 GB/s Scl | ⏭️ SKIP
▶ B3_M7_3 (64x384x384)   | 16x15   6.98 KerGOPs [min:6.4, p5:6.7, p95:12.4, max:12.8, iqr:0.4, s:1.6, CV:23%, mx/md:1.83] |  0.5 GB/s Pk |  0.8 GB/s Scl | ⏭️ SKIP
▶ B3_M7_4 (64x384x384)   | 16x15   7.17 KerGOPs [min:6.6, p5:6.9, p95:12.6, max:12.7, iqr:0.3, s:1.8, CV:25%, mx/md:1.77] |  0.5 GB/s Pk |  0.5 GB/s Scl | ⏭️ SKIP
▶ B3_M7_5 (64x384x384)   | 16x15   7.07 KerGOPs [min:6.4, p5:6.8, p95:12.3, max:12.7, iqr:0.3, s:1.7, CV:24%, mx/md:1.80] |  0.5 GB/s Pk |  0.6 GB/s Scl | ⏭️ SKIP
▶ B3_M7_6 (64x384x384)   | 16x15   7.21 KerGOPs [min:6.4, p5:6.9, p95:12.6, max:12.7, iqr:0.3, s:1.9, CV:26%, mx/md:1.76] |  0.5 GB/s Pk |  0.7 GB/s Scl | ⏭️ SKIP
▶ B3_M7_7 (64x384x384)   | 16x15   7.09 KerGOPs [min:6.2, p5:6.8, p95:12.3, max:12.7, iqr:0.2, s:1.4, CV:20%, mx/md:1.79] |  0.5 GB/s Pk |  0.7 GB/s Scl | ⏭️ SKIP
▶ B3_M7_8 (64x384x384)   | 16x15   7.33 KerGOPs [min:6.7, p5:7.0, p95:12.3, max:12.7, iqr:0.3, s:1.5, CV:20%, mx/md:1.74] |  0.5 GB/s Pk |  0.6 GB/s Scl | ⏭️ SKIP
▶ B3_M7_9 (64x384x384)   | 16x15   7.13 KerGOPs [min:6.4, p5:6.8, p95:12.5, max:12.7, iqr:2.3, s:1.9, CV:27%, mx/md:1.78] |  0.5 GB/s Pk |  0.9 GB/s Scl | ⏭️ SKIP
▶ B3_M7_10 (64x384x384)  | 16x15   7.15 KerGOPs [min:6.4, p5:6.8, p95:12.6, max:12.8, iqr:0.3, s:1.9, CV:27%, mx/md:1.79] |  0.5 GB/s Pk |  0.5 GB/s Scl | ⏭️ SKIP
▶ B2_M5_2 (384x384x64)   | 16x15  12.95 KerGOPs [min:12.2, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.00] |  2.1 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_3 (384x384x64)   | 16x15  12.94 KerGOPs [min:12.2, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.01] |  2.0 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_4 (384x384x64)   | 16x15  12.95 KerGOPs [min:12.2, p5:12.8, p95:13.0, max:13.0, iqr:0.1, s:0.1, CV:1%, mx/md:1.00] |  2.0 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_5 (384x384x64)   | 16x15  12.93 KerGOPs [min:12.2, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.01] |  2.4 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_6 (384x384x64)   | 16x15  12.97 KerGOPs [min:12.1, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.00] |  2.5 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_7 (384x384x64)   | 16x15  12.92 KerGOPs [min:5.4, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.3, CV:2%, mx/md:1.01] |  2.5 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_8 (384x384x64)   | 16x15  12.98 KerGOPs [min:12.3, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.00] |  2.5 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_9 (384x384x64)   | 16x15  12.90 KerGOPs [min:12.2, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.01] |  2.5 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
▶ B2_M5_10 (384x384x64)  | 16x15  12.95 KerGOPs [min:12.1, p5:12.7, p95:13.0, max:13.0, iqr:0.2, s:0.1, CV:1%, mx/md:1.00] |  2.1 GB/s Pk |  1.4 GB/s Scl | ⏭️ SKIP
Connection to jbpi2 closed.

Here is the recreated exo also observate unstablioty 
gcc -O3 -march=rv64gcv -mabi=lp64d -fno-strict-aliasing -DFP32 -DFAMILY_EXO -DRVV_EXO -DMR=16 -DNR=15 -I./ -I./src -I./modelLevel -o test_gemm src/test_gemm.o src/gemm_blis.o src/inutils.o src/sutils.o modelLevel/model_level.o kernels_RVV_16x16_fp32.o exo_matrix_RVV_fp32.o -lm
Connection to jbpi closed.

  Layer (M x N x K)        | Median GOPs  [min, p5, med, p95, max | iqr, std, CV%, mx/md]  Status
  -------------------------+------------------------------------------------------------------------
  L1 (1x1x1)               |   0.01 GOPs  [ 0.00,  0.01,  0.01,  0.01,  0.01 | 0.001, 0.000,   7%, 1.15]  PASS (mc:1 nc:1 kc:1)
  L2 (3x5x7)               |   0.36 GOPs  [ 0.01,  0.36,  0.36,  0.39,  0.39 | 0.027, 0.018,   5%, 1.08]  PASS (mc:3 nc:5 kc:7)
  L3 (8x8x8)               |   1.17 GOPs  [ 0.02,  1.17,  1.17,  1.23,  1.23 | 0.000, 0.041,   4%, 1.05]  PASS (mc:8 nc:8 kc:8)
  L4 (256x8x32)            |   3.01 GOPs  [ 0.99,  2.99,  3.01,  3.03,  3.05 | 0.017, 0.108,   4%, 1.01]  PASS (mc:256 nc:8 kc:32)
  L5 (8x256x32)            |   2.33 GOPs  [ 1.49,  2.23,  2.33,  2.35,  2.40 | 0.026, 0.070,   3%, 1.03]  PASS (mc:8 nc:256 kc:32)
  L6 (64x64x300)           |   6.99 GOPs  [ 6.09,  6.79,  6.99,  7.01,  7.03 | 0.034, 0.084,   1%, 1.01]  PASS (mc:64 nc:64 kc:128)
  L7 (128x128x128)         |   9.85 GOPs  [ 8.43,  9.55,  9.85,  9.89,  9.92 | 0.048, 0.128,   1%, 1.01]  PASS (mc:128 nc:128 kc:128)
  B3_M7_1 (64x384x384)     |   5.44 GOPs  [ 4.87,  5.08,  5.44,  8.03,  8.14 | 2.630, 1.290,  24%, 1.50]  PASS (mc:64 nc:384 kc:128)
  B3_M7_2 (64x384x384)     |   7.76 GOPs  [ 4.93,  5.24,  7.76,  8.03,  8.14 | 2.416, 1.113,  14%, 1.05]  PASS (mc:64 nc:384 kc:128)
  B3_M7_3 (64x384x384)     |   6.86 GOPs  [ 3.87,  5.21,  6.86,  8.06,  8.15 | 2.431, 1.126,  16%, 1.19]  PASS (mc:64 nc:384 kc:128)
  B3_M7_4 (64x384x384)     |   7.73 GOPs  [ 5.04,  5.22,  7.73,  8.05,  8.13 | 1.498, 1.084,  14%, 1.05]  PASS (mc:64 nc:384 kc:128)
  B3_M7_5 (64x384x384)     |   7.83 GOPs  [ 5.01,  5.29,  7.83,  8.08,  8.17 | 1.190, 0.996,  13%, 1.04]  PASS (mc:64 nc:384 kc:128)
  B3_M7_6 (64x384x384)     |   7.69 GOPs  [ 5.08,  5.34,  7.69,  8.05,  8.16 | 0.400, 0.859,  11%, 1.06]  PASS (mc:64 nc:384 kc:128)
  B3_M7_7 (64x384x384)     |   7.79 GOPs  [ 5.01,  5.28,  7.79,  8.06,  8.14 | 1.260, 1.028,  13%, 1.05]  PASS (mc:64 nc:384 kc:128)
  B3_M7_8 (64x384x384)     |   7.87 GOPs  [ 6.35,  6.67,  7.87,  8.07,  8.14 | 0.290, 0.455,   6%, 1.03]  PASS (mc:64 nc:384 kc:128)
  B3_M7_9 (64x384x384)     |   7.76 GOPs  [ 5.08,  5.29,  7.76,  8.03,  8.15 | 2.437, 1.193,  15%, 1.05]  PASS (mc:64 nc:384 kc:128)
  B3_M7_10 (64x384x384)    |   7.90 GOPs  [ 4.99,  5.39,  7.90,  8.08,  8.19 | 0.201, 0.764,  10%, 1.04]  PASS (mc:64 nc:384 kc:128)
  B2_M5_2 (384x384x64)     |  11.92 GOPs  [11.31, 11.76, 11.92, 12.00, 12.04 | 0.115, 0.087,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_3 (384x384x64)     |  11.95 GOPs  [11.49, 11.81, 11.95, 12.03, 12.08 | 0.112, 0.074,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_4 (384x384x64)     |  11.94 GOPs  [11.28, 11.79, 11.94, 12.02, 12.07 | 0.109, 0.087,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_5 (384x384x64)     |  12.03 GOPs  [11.24, 11.88, 12.03, 12.12, 12.19 | 0.115, 0.090,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_6 (384x384x64)     |  12.02 GOPs  [11.37, 11.86, 12.02, 12.10, 12.14 | 0.109, 0.081,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_7 (384x384x64)     |  12.08 GOPs  [11.49, 11.92, 12.08, 12.15, 12.21 | 0.123, 0.084,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_8 (384x384x64)     |  12.08 GOPs  [10.96, 11.91, 12.08, 12.16, 12.20 | 0.132, 0.094,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_9 (384x384x64)     |  12.00 GOPs  [11.34, 11.85, 12.00, 12.10, 12.13 | 0.116, 0.085,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  B2_M5_10 (384x384x64)    |  12.06 GOPs  [11.52, 11.90, 12.06, 12.13, 12.17 | 0.128, 0.080,   1%, 1.01]  PASS (mc:384 nc:384 kc:64)
  
  
  Same design, ref exo design recompile and run again show more stable 
  B3_M7_4 (64x384x384)     |   7.82 GOPs  [ 6.44,  7.51,  7.82,  7.98,  8.08 | 0.157, 0.164,   2%, 1.03]  PASS (mc:64 nc:384 kc:128)
  B3_M7_5 (64x384x384)     |   7.82 GOPs  [ 6.97,  7.65,  7.82,  7.99,  8.08 | 0.131, 0.105,   1%, 1.03]  PASS (mc:64 nc:384 kc:128)
  B3_M7_6 (64x384x384)     |   7.85 GOPs  [ 6.97,  7.69,  7.85,  8.00,  8.06 | 0.131, 0.100,   1%, 1.03]  PASS (mc:64 nc:384 kc:128)
  B3_M7_7 (64x384x384)     |   7.83 GOPs  [ 7.08,  7.67,  7.83,  7.98,  8.04 | 0.131, 0.100,   1%, 1.03]  PASS (mc:64 nc:384 kc:128)

```

</aside>

- Hyposisite of KC issue, not stand, KC handle correctl;y
    
    ```jsx
    Listed directory gemm_blis_family
    Listed directory src
    Viewed gemm_blis.c:140-250
    Viewed gemm_blis.c:1-140
    Viewed gemm_dispatch_xdsl.cpp:30-60
    
    Yes, the reference design absolutely has `Kc` blocking! 
    
    If you look inside the reference code (`ext/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c`), you will find the classic 5-loop BLIS macrokernel. It uses explicitly defined cache parameters (`MC`, `NC`, and `KC`) mapped to the hardware's L1/L2/L3 cache sizes:
    
    ```c
      for ( jc=0; jc<n; jc+=NC ) {         // L3 Cache block (N)
        for ( pc=0; pc<k; pc+=KC ) {       // L2 Cache block (K) -> This prevents thrashing!
          // pack B...
          for ( ic=0; ic<m; ic+=MC ) {     // L1 Cache block (M)
            // pack A...
            for ( jr=0; jr<nc; jr+=NR ) {  // Register block (NR)
              for ( ir=0; ir<mc; ir+=MR ) { // Register block (MR)
                 // ... call microkernel ...
    ```
    
    ### The Irony of your `pure_xdsl` design:
    Your `pure_xdsl` dispatcher (`gemm_dispatch_xdsl.cpp`) **actually has the `Kc` loop built into it!** 
    
    However, if you look inside your C++ benchmark driver (`driver_benchmark.cpp`), you will see this function at the top:
    ```cpp
    void get_optim_mc_nc_kc(int elem_size, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc) {
        *mc = ((m + mr - 1) / mr) * mr;
        *nc = ((n + nr - 1) / nr) * nr;
        *kc = k;  // <--- HERE IS THE PROBLEM!
    }
    ```
    Your driver function is literally hardcoding `kc = k`. 
    
    By setting `kc = k`, you are bypassing the cache blocking loop in the dispatcher entirely. It forces the macrokernel to chew through the entire `K=384` dimension in a single pass, which is exactly what wipes out the L1/L2 cache and destroys your stability.
    
    If you change `*kc = k;` to something like `*kc = 128;` (or whatever optimal chunk size fits in the L2 cache of the `jbpi2` architecture), your `16x15` median performance on `B3_M7` will immediately shoot up and stabilize!`
    ```
    

- Hyposisite of temp_c ruined, not stand it not affect

<aside>
⚠️

the project is in stand still, the 16x15 kernel is unstable in terms of performance, sometimes fast sometimes no, kernel performance affectred by the dimenttion of the problem, 

if the problem is b3_m3 stability manitain, if its b3_m7 stablity is poor,

i want rerun ref design in a dir, adapotingn into a new project, instead of run it natively at its project, trying to fin dout where is the problem come from. 

The perf is similar but unstavble 

</aside>

Evalute, the time, GFLOP,

- fix  the dimention of kernel, evaluate the perf of GEMM (uK + Packing), vector packing  and scaler packing
    - Find out the affetc of packing

For a given problem, evalu the perfornmace combined with packing and kernel,

Compltet axby for Fp32 and int8i8i32, 

GHW dialect intergration, create a GHW dialect that will trick mlitr_tranlaste to gene4rate CPP file that intergrate GHW,
