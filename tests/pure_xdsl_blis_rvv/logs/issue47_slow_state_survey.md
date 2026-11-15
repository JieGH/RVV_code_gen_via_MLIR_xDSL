# Slow state across every shape (issue #47): 796 case-launches from 193 run folders up to 2026-09-28_235959

Slow state = at least 5% of a launch's reps slow (reps_report.py's split). Speeds from uninstrumented reference-protocol logs on a pinned core other than 0 only.

## 1. Inventory

| board | build | protocol | core | packer | nc override | counts? | per-rep files | layers seen |
|---|---|---|---|---|---|---|---|---|
| jbpi | instrumented | warm-up | 2 | vector | ignored | no: instrumented build (counts only) | 3 | B2, B3, B3_K256, R24 |
| jbpi | instrumented | warm-up | 4 | scalar | shipped | no: instrumented build (counts only) | 3 | B2, B3, B3_K256, B3_N320, B3_nc128, B3_nc384, R24_nc128, R24_nc384 |
| jbpi | instrumented | warm-up | 4 | scalar | ignored | no: instrumented build (counts only) | 3 | B2, B3, B3_K256, R24 |
| jbpi | instrumented | warm-up | 4 | vector | shipped | no: instrumented build (counts only) | 24 | B2, B3, B3_K256, B3_N320, B3_nc128, B3_nc384, R24, R24_nc128, R24_nc384 |
| jbpi | instrumented | warm-up | 4 | vector | ignored | no: instrumented build (counts only) | 4 | B2, B3, B3_K256, R24 |
| jbpi | instrumented | warm-up | 4 | vector+zeroline | shipped | no: instrumented build (counts only) | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi | uninstrumented | reference | 4 | scalar | shipped | yes | 3 | B2_nc128, B2_nc384, B3_K256, B3_K256_nc128, B3_N192, B3_N192_nc128, B3_N256, B3_N256_nc128, B3_N320, B3_N320_nc128, B3_nc120, B3_nc128, B3_nc192, B3_nc240, B3_nc256, B3_nc32, B3_nc320, B3_nc375, B3_nc384, B3_nc64, B3_nc96, R12_nc120, R12_nc128, R12_nc192, R12_nc240, R12_nc256, R12_nc32, R12_nc320, R12_nc375, R12_nc384, R12_nc64, R12_nc96, R24_nc120, R24_nc128, R24_nc192, R24_nc240, R24_nc256, R24_nc32, R24_nc320, R24_nc375, R24_nc384, R24_nc64, R24_nc96, R8_nc120, R8_nc128, R8_nc192, R8_nc240, R8_nc256, R8_nc32, R8_nc320, R8_nc375, R8_nc384, R8_nc64, R8_nc96 |
| jbpi | uninstrumented | reference | 4 | vector | shipped | yes | 24 | B2, B2_nc128, B2_nc384, B3, B3_K256, B3_K256_nc128, B3_N192, B3_N192_nc128, B3_N256, B3_N256_nc128, B3_N320, B3_N320_nc128, B3_nc120, B3_nc128, B3_nc192, B3_nc240, B3_nc256, B3_nc32, B3_nc320, B3_nc375, B3_nc384, B3_nc64, B3_nc96, R12_nc120, R12_nc128, R12_nc192, R12_nc240, R12_nc256, R12_nc32, R12_nc320, R12_nc375, R12_nc384, R12_nc64, R12_nc96, R24, R24_nc120, R24_nc128, R24_nc192, R24_nc240, R24_nc256, R24_nc32, R24_nc320, R24_nc375, R24_nc384, R24_nc64, R24_nc96, R8_nc120, R8_nc128, R8_nc192, R8_nc240, R8_nc256, R8_nc32, R8_nc320, R8_nc375, R8_nc384, R8_nc64, R8_nc96 |
| jbpi | uninstrumented | reference | 4 | vector+zeroline | shipped | yes | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi2 | instrumented | warm-up | 4 | vector | shipped | no: instrumented build (counts only) | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi2 | instrumented | warm-up | 4 | vector+zeroline | shipped | no: instrumented build (counts only) | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi2 | uninstrumented | reference | 4 | vector | shipped | yes | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi2 | uninstrumented | reference | 4 | vector | ignored | yes | 14 | B1, B4, B5, S1, S2, S3, S4, S5 |
| jbpi2 | uninstrumented | reference | 4 | vector+zeroline | shipped | yes | 21 | B2, B3, B3_K256, B3_nc128, R24 |

One row per per-rep file:

| run folder | board | build | protocol | core | packer | tile | counts? |
|---|---|---|---|---|---|---|---|
| run_2026-09-27_104338 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_105417 | jbpi | uninstrumented | reference | 4 | scalar | 16x15 | yes |
| run_2026-09-27_110456 | jbpi | uninstrumented | reference | 4 | scalar | 16x15 | yes |
| run_2026-09-27_111536 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_112614 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_113653 | jbpi | uninstrumented | reference | 4 | scalar | 16x15 | yes |
| run_2026-09-27_114341 | jbpi | instrumented | warm-up | 4 | vector | 16x14 | no: instrumented build (counts only) |
| run_2026-09-27_114341 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_114341 | jbpi | instrumented | warm-up | 4 | vector | 16x16 | no: instrumented build (counts only) |
| run_2026-09-27_115023 | jbpi | instrumented | warm-up | 4 | scalar | 16x14 | no: instrumented build (counts only) |
| run_2026-09-27_115023 | jbpi | instrumented | warm-up | 4 | scalar | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_115023 | jbpi | instrumented | warm-up | 4 | scalar | 16x16 | no: instrumented build (counts only) |
| run_2026-09-27_131302 | jbpi | instrumented | warm-up | 4 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_131537 | jbpi | instrumented | warm-up | 4 | scalar (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_131815 | jbpi | instrumented | warm-up | 2 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_132052 | jbpi | instrumented | warm-up | 2 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_132328 | jbpi | instrumented | warm-up | 4 | scalar (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_132606 | jbpi | instrumented | warm-up | 4 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_132843 | jbpi | instrumented | warm-up | 4 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_133117 | jbpi | instrumented | warm-up | 4 | scalar (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_133354 | jbpi | instrumented | warm-up | 2 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_133825 | jbpi | instrumented | warm-up | 4 | vector (nc ignored) | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_152131 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_152415 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_152700 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_152944 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_153227 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_153512 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_153743 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_154015 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_154247 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_154519 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_154752 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_155023 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_161022 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_161307 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_161552 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_161837 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_162122 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_162407 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_162639 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_162910 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_163142 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_163413 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_163646 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_163917 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_173243 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_173527 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_173812 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_174055 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_174339 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_174624 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_174856 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_175128 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_175400 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_175633 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_175905 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_180137 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_180420 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_180705 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_180949 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_181232 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_181516 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_181800 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_182032 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_182304 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_182535 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_182807 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_183039 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_183311 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_212934 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_213219 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_213504 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_213749 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_214033 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_214315 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_214548 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_214820 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_215052 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_215323 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_215555 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_215827 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_220933 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_221217 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_221502 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_221746 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_222029 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_222314 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_222546 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_222818 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_223050 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_223322 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_223554 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_223825 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_224933 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_225219 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_225503 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_225747 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_230032 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_230317 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_230549 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_230821 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_231053 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_231325 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_231557 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_231828 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_232934 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_233220 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_233504 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_233749 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_234031 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_234315 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-27_234547 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_234818 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_235050 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-27_235322 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_235553 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-27_235825 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_000934 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_001219 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_001503 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_001747 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_002031 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_002316 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_002548 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_002820 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_003052 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_003323 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_003555 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_003827 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_004934 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_005217 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_005508 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_005751 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_010035 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_010320 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_010551 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_010823 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_011055 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_011327 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_011558 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_011830 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_012935 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_013220 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_013504 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_013748 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_014033 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_014317 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_014549 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_014820 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_015052 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_015324 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_015556 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_015828 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_020935 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_021218 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_021503 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_021747 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_022030 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_022315 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_022547 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_022818 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_023050 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_023322 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_023554 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_023826 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_024934 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_025219 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_025503 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_025747 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_030031 | jbpi | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_030316 | jbpi | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_030547 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_030819 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_031051 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_031323 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_031555 | jbpi | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_031827 | jbpi | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_032934 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_033219 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_033504 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_033748 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_034032 | jbpi2 | instrumented | warm-up | 4 | vector | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_034315 | jbpi2 | instrumented | warm-up | 4 | vector+zeroline | 16x15 | no: instrumented build (counts only) |
| run_2026-09-28_034547 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_034819 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_035050 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_035322 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_035554 | jbpi2 | uninstrumented | reference | 4 | vector | 16x15 | yes |
| run_2026-09-28_035826 | jbpi2 | uninstrumented | reference | 4 | vector+zeroline | 16x15 | yes |
| run_2026-09-28_113634 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_113634 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |
| run_2026-09-28_114331 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_114331 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |
| run_2026-09-28_115034 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_115034 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |
| run_2026-09-28_121056 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_121056 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |
| run_2026-09-28_121503 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_121503 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |
| run_2026-09-28_121903 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_121903 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |
| run_2026-09-28_122304 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x6 | yes |
| run_2026-09-28_122304 | jbpi2 | uninstrumented | reference | 4 | vector (nc ignored) | 32x7 | yes |

## 2. Slow state per case

| board | packer | tile | case | nc | launches (with a slow state) | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % (avg, lo-hi) | fast start s (avg) | fast start reps (avg) | clumping | CV % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jbpi | vector | 16x15 | B2 (384x384x64) | 384 | 21 (0) | 12.40 (12.18-12.55) | 12.48 | 0 (0-0) | - | - | - | 2.0 |
| jbpi | vector+zeroline | 16x15 | B2 (384x384x64) | 384 | 21 (0) | 12.23 (11.91-12.47) | 12.38 | 0 (0-0) | - | - | - | 3.1 |
| jbpi | scalar | 16x15 | B2_nc128 (384x384x64) | 128 | 3 (0) | 10.51 (10.46-10.60) | 10.53 | 0 (0-0) | - | - | - | 1.4 |
| jbpi | vector | 16x15 | B2_nc128 (384x384x64) | 128 | 3 (0) | 11.45 (11.42-11.51) | 11.49 | 0 (0-0) | - | - | - | 1.0 |
| jbpi | scalar | 16x15 | B2_nc384 (384x384x64) | 384 | 3 (0) | 11.82 (11.61-11.93) | 11.90 | 0 (0-0) | - | - | - | 2.1 |
| jbpi | vector | 16x15 | B2_nc384 (384x384x64) | 384 | 3 (0) | 11.98 (11.90-12.07) | 12.37 | 0 (0-0) | - | - | - | 6.2 |
| jbpi | vector | 16x15 | B3 (64x384x384) | 384 | 21 (21) | 4.49 (4.34-5.08) | 4.28 | 89 (77-95) | 0.06 | 21 | 0.06 | 21.1 |
| jbpi | vector+zeroline | 16x15 | B3 (64x384x384) | 384 | 21 (21) | 4.47 (4.29-5.12) | 4.24 | 88 (76-97) | 0.04 | 17 | 0.06 | 21.2 |
| jbpi | scalar | 16x15 | B3_K256 (64x384x256) | 384 | 3 (3) | 4.54 (4.49-4.62) | 4.39 | 93 (91-95) | 0.05 | 32 | 0.05 | 20.8 |
| jbpi | vector | 16x15 | B3_K256 (64x384x256) | 384 | 24 (24) | 4.72 (4.52-5.10) | 4.60 | 95 (79-98) | 0.02 | 16 | 0.08 | 18.2 |
| jbpi | vector+zeroline | 16x15 | B3_K256 (64x384x256) | 384 | 21 (21) | 4.68 (4.54-5.22) | 4.56 | 95 (80-97) | 0.02 | 14 | 0.08 | 17.6 |
| jbpi | scalar | 16x15 | B3_K256_nc128 (64x384x256) | 128 | 3 (3) | 5.64 (5.53-5.78) | 5.25 | 74 (72-76) | 0.33 | 209 | 0.01 | 22.2 |
| jbpi | vector | 16x15 | B3_K256_nc128 (64x384x256) | 128 | 3 (3) | 5.69 (5.45-6.16) | 5.44 | 89 (74-97) | 0.02 | 15 | 0.05 | 15.1 |
| jbpi | scalar | 16x15 | B3_N192 (64x192x384) | 192 | 3 (2) | 7.72 (7.37-7.94) | 8.17 | 17 (0-40) | 1.61 | 1422 | 0.15 | 10.5 |
| jbpi | vector | 16x15 | B3_N192 (64x192x384) | 192 | 3 (3) | 6.09 (6.07-6.10) | 5.58 | 80 (78-82) | 0.15 | 143 | 0.01 | 22.0 |
| jbpi | scalar | 16x15 | B3_N192_nc128 (64x192x384) | 128 | 3 (0) | 7.47 (7.42-7.53) | 7.49 | 0 (0-0) | - | - | - | 1.9 |
| jbpi | vector | 16x15 | B3_N192_nc128 (64x192x384) | 128 | 3 (3) | 6.19 (6.01-6.32) | 5.89 | 66 (53-76) | 0.40 | 341 | 0.01 | 19.7 |
| jbpi | scalar | 16x15 | B3_N256 (64x256x384) | 256 | 3 (3) | 7.48 (6.96-7.83) | 8.11 | 16 (11-23) | 2.55 | 1652 | 0.09 | 13.6 |
| jbpi | vector | 16x15 | B3_N256 (64x256x384) | 256 | 3 (3) | 5.36 (5.17-5.46) | 4.85 | 83 (80-88) | 0.18 | 118 | 0.02 | 23.8 |
| jbpi | scalar | 16x15 | B3_N256_nc128 (64x256x384) | 128 | 3 (0) | 7.54 (7.49-7.63) | 7.73 | 0 (0-0) | - | - | - | 6.0 |
| jbpi | vector | 16x15 | B3_N256_nc128 (64x256x384) | 128 | 3 (3) | 5.93 (5.56-6.60) | 6.16 | 68 (43-80) | 0.19 | 124 | 0.01 | 22.6 |
| jbpi | scalar | 16x15 | B3_N320 (64x320x384) | 320 | 3 (3) | 6.99 (6.63-7.17) | 7.87 | 32 (28-35) | 1.04 | 537 | 0.04 | 18.0 |
| jbpi | vector | 16x15 | B3_N320 (64x320x384) | 320 | 3 (3) | 5.19 (4.85-5.53) | 4.51 | 75 (65-81) | 0.22 | 118 | 0.02 | 30.9 |
| jbpi | scalar | 16x15 | B3_N320_nc128 (64x320x384) | 128 | 3 (1) | 7.22 (7.08-7.32) | 7.38 | 3 (0-10) | 1.73 | 817 | 0.05 | 5.8 |
| jbpi | vector | 16x15 | B3_N320_nc128 (64x320x384) | 128 | 3 (3) | 5.90 (5.55-6.61) | 5.92 | 61 (35-77) | 0.27 | 134 | 0.01 | 23.4 |
| jbpi | scalar | 16x15 | B3_nc120 (64x384x384) | 120 | 3 (2) | 7.26 (7.05-7.41) | 7.62 | 10 (0-21) | 3.81 | 1564 | 0.02 | 9.4 |
| jbpi | vector | 16x15 | B3_nc120 (64x384x384) | 120 | 3 (3) | 5.67 (5.45-5.85) | 5.31 | 77 (69-86) | 0.22 | 95 | 0.01 | 23.6 |
| jbpi | scalar | 16x15 | B3_nc128 (64x384x384) | 128 | 3 (1) | 7.36 (7.31-7.43) | 7.58 | 6 (0-18) | 4.69 | 1915 | 0.05 | 7.1 |
| jbpi | vector | 16x15 | B3_nc128 (64x384x384) | 128 | 24 (24) | 5.68 (5.26-6.82) | 5.39 | 73 (42-88) | 0.16 | 66 | 0.03 | 23.8 |
| jbpi | vector+zeroline | 16x15 | B3_nc128 (64x384x384) | 128 | 21 (21) | 6.04 (5.51-7.36) | 6.20 | 60 (32-73) | 0.42 | 181 | 0.02 | 24.6 |
| jbpi | scalar | 16x15 | B3_nc192 (64x384x384) | 192 | 3 (3) | 7.04 (6.68-7.65) | 7.48 | 37 (24-47) | 0.86 | 371 | 0.09 | 15.0 |
| jbpi | vector | 16x15 | B3_nc192 (64x384x384) | 192 | 3 (3) | 5.55 (5.44-5.69) | 4.83 | 76 (74-78) | 0.20 | 93 | 0.01 | 27.4 |
| jbpi | scalar | 16x15 | B3_nc240 (64x384x384) | 240 | 3 (3) | 6.81 (6.53-7.04) | 7.46 | 40 (36-44) | 1.57 | 665 | 0.05 | 16.9 |
| jbpi | vector | 16x15 | B3_nc240 (64x384x384) | 240 | 3 (3) | 5.25 (5.23-5.28) | 4.76 | 82 (82-83) | 0.26 | 117 | 0.02 | 27.1 |
| jbpi | scalar | 16x15 | B3_nc256 (64x384x384) | 256 | 3 (3) | 7.07 (6.57-7.54) | 7.63 | 29 (12-46) | 2.15 | 888 | 0.04 | 13.3 |
| jbpi | vector | 16x15 | B3_nc256 (64x384x384) | 256 | 3 (3) | 5.10 (5.04-5.18) | 4.55 | 79 (77-81) | 0.18 | 78 | 0.01 | 26.9 |
| jbpi | scalar | 16x15 | B3_nc32 (64x384x384) | 32 | 3 (0) | 5.31 (5.21-5.48) | 5.31 | 0 (0-0) | - | - | - | 0.6 |
| jbpi | vector | 16x15 | B3_nc32 (64x384x384) | 32 | 3 (0) | 6.53 (6.43-6.59) | 6.53 | 0 (0-0) | - | - | - | 0.9 |
| jbpi | scalar | 16x15 | B3_nc320 (64x384x384) | 320 | 3 (2) | 7.14 (6.75-7.76) | 7.82 | 23 (4-38) | 1.03 | 426 | 0.04 | 13.8 |
| jbpi | vector | 16x15 | B3_nc320 (64x384x384) | 320 | 3 (3) | 4.89 (4.65-5.07) | 4.34 | 81 (76-89) | 0.14 | 58 | 0.02 | 28.4 |
| jbpi | scalar | 16x15 | B3_nc375 (64x384x384) | 375 | 3 (3) | 6.81 (6.71-6.90) | 7.63 | 38 (32-43) | 2.18 | 942 | 0.03 | 18.1 |
| jbpi | vector | 16x15 | B3_nc375 (64x384x384) | 375 | 3 (3) | 4.92 (4.80-4.99) | 4.21 | 77 (75-78) | 0.11 | 49 | 0.02 | 33.4 |
| jbpi | scalar | 16x15 | B3_nc384 (64x384x384) | 384 | 3 (3) | 6.65 (6.36-7.06) | 7.21 | 46 (42-51) | 0.49 | 215 | 0.03 | 21.0 |
| jbpi | vector | 16x15 | B3_nc384 (64x384x384) | 384 | 3 (3) | 4.73 (4.65-4.79) | 4.27 | 86 (84-89) | 0.14 | 64 | 0.02 | 30.7 |
| jbpi | scalar | 16x15 | B3_nc64 (64x384x384) | 64 | 3 (0) | 6.65 (6.60-6.74) | 6.65 | 0 (0-0) | - | - | - | 1.1 |
| jbpi | vector | 16x15 | B3_nc64 (64x384x384) | 64 | 3 (3) | 6.71 (6.50-7.01) | 7.55 | 22 (14-27) | 1.94 | 766 | 0.01 | 16.0 |
| jbpi | scalar | 16x15 | B3_nc96 (64x384x384) | 96 | 3 (0) | 7.27 (7.16-7.42) | 7.28 | 0 (0-0) | - | - | - | 1.6 |
| jbpi | vector | 16x15 | B3_nc96 (64x384x384) | 96 | 3 (3) | 5.81 (5.48-6.34) | 5.59 | 66 (46-79) | 0.37 | 151 | 0.01 | 22.4 |
| jbpi | scalar | 16x15 | R12_nc120 (192x384x384) | 120 | 3 (0) | 8.71 (8.69-8.75) | 8.74 | 0 (0-0) | - | - | - | 1.4 |
| jbpi | vector | 16x15 | R12_nc120 (192x384x384) | 120 | 3 (3) | 7.70 (7.64-7.82) | 8.55 | 45 (43-47) | 1.02 | 168 | 0.02 | 18.8 |
| jbpi | scalar | 16x15 | R12_nc128 (192x384x384) | 128 | 3 (0) | 8.82 (8.81-8.83) | 8.83 | 0 (0-0) | - | - | - | 0.8 |
| jbpi | vector | 16x15 | R12_nc128 (192x384x384) | 128 | 3 (3) | 7.53 (7.26-7.72) | 8.11 | 47 (41-57) | 1.01 | 162 | 0.02 | 19.5 |
| jbpi | scalar | 16x15 | R12_nc192 (192x384x384) | 192 | 3 (0) | 9.30 (9.13-9.39) | 9.37 | 0 (0-0) | - | - | - | 2.3 |
| jbpi | vector | 16x15 | R12_nc192 (192x384x384) | 192 | 3 (3) | 7.40 (7.35-7.44) | 7.61 | 54 (54-54) | 0.37 | 63 | 0.02 | 23.3 |
| jbpi | scalar | 16x15 | R12_nc240 (192x384x384) | 240 | 3 (0) | 9.26 (9.20-9.32) | 9.39 | 1 (0-4) | - | - | - | 3.8 |
| jbpi | vector | 16x15 | R12_nc240 (192x384x384) | 240 | 3 (3) | 7.91 (7.36-8.47) | 8.10 | 47 (28-59) | 0.53 | 91 | 0.02 | 18.7 |
| jbpi | scalar | 16x15 | R12_nc256 (192x384x384) | 256 | 3 (0) | 9.11 (9.08-9.13) | 9.15 | 0 (0-0) | - | - | - | 1.6 |
| jbpi | vector | 16x15 | R12_nc256 (192x384x384) | 256 | 3 (3) | 7.28 (7.06-7.43) | 7.37 | 55 (51-57) | 0.42 | 68 | 0.02 | 21.4 |
| jbpi | scalar | 16x15 | R12_nc32 (192x384x384) | 32 | 3 (0) | 5.70 (5.66-5.74) | 5.71 | 0 (0-0) | - | - | - | 0.4 |
| jbpi | vector | 16x15 | R12_nc32 (192x384x384) | 32 | 3 (0) | 6.59 (6.53-6.64) | 6.59 | 0 (0-0) | - | - | - | 0.7 |
| jbpi | scalar | 16x15 | R12_nc320 (192x384x384) | 320 | 3 (0) | 9.02 (8.93-9.08) | 9.09 | 2 (0-5) | - | - | - | 2.6 |
| jbpi | vector | 16x15 | R12_nc320 (192x384x384) | 320 | 3 (3) | 7.49 (6.97-7.84) | 8.12 | 49 (36-62) | 0.42 | 68 | 0.02 | 19.9 |
| jbpi | scalar | 16x15 | R12_nc375 (192x384x384) | 375 | 3 (0) | 9.31 (9.28-9.34) | 9.35 | 0 (0-0) | - | - | - | 1.8 |
| jbpi | vector | 16x15 | R12_nc375 (192x384x384) | 375 | 3 (3) | 6.80 (6.58-7.00) | 6.71 | 68 (58-74) | 0.56 | 94 | 0.03 | 23.6 |
| jbpi | scalar | 16x15 | R12_nc384 (192x384x384) | 384 | 3 (2) | 9.23 (9.01-9.39) | 9.72 | 7 (0-14) | 4.29 | 730 | 0.02 | 9.8 |
| jbpi | vector | 16x15 | R12_nc384 (192x384x384) | 384 | 3 (3) | 7.29 (7.21-7.41) | 7.58 | 64 (58-68) | 0.47 | 82 | 0.03 | 22.3 |
| jbpi | scalar | 16x15 | R12_nc64 (192x384x384) | 64 | 3 (0) | 7.47 (7.44-7.49) | 7.47 | 0 (0-0) | - | - | - | 0.5 |
| jbpi | vector | 16x15 | R12_nc64 (192x384x384) | 64 | 3 (1) | 8.14 (8.03-8.28) | 8.30 | 3 (0-9) | 0.01 | 1 | 0.02 | 4.2 |
| jbpi | scalar | 16x15 | R12_nc96 (192x384x384) | 96 | 3 (0) | 8.36 (8.32-8.40) | 8.37 | 0 (0-0) | - | - | - | 0.7 |
| jbpi | vector | 16x15 | R12_nc96 (192x384x384) | 96 | 3 (2) | 8.55 (8.17-8.96) | 9.03 | 14 (2-23) | 1.21 | 190 | 0.02 | 9.1 |
| jbpi | vector | 16x15 | R24 (384x384x384) | 384 | 21 (21) | 8.02 (7.57-8.58) | 8.23 | 40 (25-64) | 1.15 | 92 | 0.03 | 13.0 |
| jbpi | vector+zeroline | 16x15 | R24 (384x384x384) | 384 | 21 (9) | 8.39 (8.00-8.89) | 8.86 | 12 (0-39) | 1.83 | 147 | 0.03 | 11.2 |
| jbpi | scalar | 16x15 | R24_nc120 (384x384x384) | 120 | 3 (0) | 8.29 (8.27-8.31) | 8.30 | 0 (0-0) | - | - | - | 0.5 |
| jbpi | vector | 16x15 | R24_nc120 (384x384x384) | 120 | 3 (0) | 8.30 (8.15-8.51) | 8.54 | 0 (0-0) | - | - | - | 5.6 |
| jbpi | scalar | 16x15 | R24_nc128 (384x384x384) | 128 | 3 (0) | 8.29 (8.26-8.31) | 8.29 | 0 (0-0) | - | - | - | 0.5 |
| jbpi | vector | 16x15 | R24_nc128 (384x384x384) | 128 | 3 (1) | 7.58 (7.43-7.66) | 7.81 | 9 (0-27) | 1.94 | 144 | 0.03 | 11.6 |
| jbpi | scalar | 16x15 | R24_nc192 (384x384x384) | 192 | 3 (0) | 8.85 (8.84-8.86) | 8.86 | 0 (0-0) | - | - | - | 0.6 |
| jbpi | vector | 16x15 | R24_nc192 (384x384x384) | 192 | 3 (2) | 8.25 (8.11-8.47) | 8.68 | 13 (0-20) | 2.77 | 216 | 0.03 | 10.1 |
| jbpi | scalar | 16x15 | R24_nc240 (384x384x384) | 240 | 3 (0) | 8.90 (8.88-8.92) | 8.91 | 0 (0-0) | - | - | - | 0.9 |
| jbpi | vector | 16x15 | R24_nc240 (384x384x384) | 240 | 3 (0) | 8.56 (8.26-8.80) | 8.94 | 0 (0-0) | - | - | - | 7.8 |
| jbpi | scalar | 16x15 | R24_nc256 (384x384x384) | 256 | 3 (0) | 8.69 (8.66-8.72) | 8.69 | 0 (0-0) | - | - | - | 0.5 |
| jbpi | vector | 16x15 | R24_nc256 (384x384x384) | 256 | 3 (1) | 8.10 (7.94-8.28) | 8.61 | 7 (0-21) | 0.59 | 44 | 0.03 | 10.2 |
| jbpi | scalar | 16x15 | R24_nc32 (384x384x384) | 32 | 3 (0) | 5.43 (5.42-5.46) | 5.43 | 0 (0-0) | - | - | - | 0.3 |
| jbpi | vector | 16x15 | R24_nc32 (384x384x384) | 32 | 3 (0) | 5.93 (5.90-5.94) | 5.94 | 0 (0-0) | - | - | - | 0.8 |
| jbpi | scalar | 16x15 | R24_nc320 (384x384x384) | 320 | 3 (0) | 8.68 (8.67-8.68) | 8.69 | 0 (0-0) | - | - | - | 0.8 |
| jbpi | vector | 16x15 | R24_nc320 (384x384x384) | 320 | 3 (2) | 8.02 (7.79-8.20) | 8.37 | 12 (0-20) | 2.51 | 188 | 0.04 | 10.3 |
| jbpi | scalar | 16x15 | R24_nc375 (384x384x384) | 375 | 3 (0) | 8.95 (8.94-8.97) | 8.96 | 0 (0-0) | - | - | - | 0.7 |
| jbpi | vector | 16x15 | R24_nc375 (384x384x384) | 375 | 3 (1) | 8.43 (7.96-8.77) | 8.77 | 8 (0-25) | 3.26 | 245 | 0.03 | 9.3 |
| jbpi | scalar | 16x15 | R24_nc384 (384x384x384) | 384 | 3 (0) | 9.29 (9.26-9.32) | 9.30 | 0 (0-0) | - | - | - | 0.7 |
| jbpi | vector | 16x15 | R24_nc384 (384x384x384) | 384 | 3 (1) | 8.53 (8.42-8.68) | 8.96 | 6 (0-18) | 2.09 | 161 | 0.04 | 10.4 |
| jbpi | scalar | 16x15 | R24_nc64 (384x384x384) | 64 | 3 (0) | 7.12 (7.10-7.15) | 7.12 | 0 (0-0) | - | - | - | 0.4 |
| jbpi | vector | 16x15 | R24_nc64 (384x384x384) | 64 | 3 (0) | 7.52 (7.50-7.55) | 7.53 | 0 (0-0) | - | - | - | 0.7 |
| jbpi | scalar | 16x15 | R24_nc96 (384x384x384) | 96 | 3 (0) | 7.95 (7.93-7.99) | 7.96 | 0 (0-0) | - | - | - | 0.5 |
| jbpi | vector | 16x15 | R24_nc96 (384x384x384) | 96 | 3 (0) | 8.01 (7.73-8.18) | 8.17 | 0 (0-0) | - | - | - | 4.8 |
| jbpi | scalar | 16x15 | R8_nc120 (128x384x384) | 120 | 3 (0) | 8.43 (8.34-8.48) | 8.55 | 0 (0-0) | - | - | - | 3.8 |
| jbpi | vector | 16x15 | R8_nc120 (128x384x384) | 120 | 3 (3) | 6.94 (6.49-7.78) | 7.64 | 57 (31-71) | 0.29 | 69 | 0.02 | 22.1 |
| jbpi | scalar | 16x15 | R8_nc128 (128x384x384) | 128 | 3 (0) | 8.39 (8.28-8.54) | 8.56 | 0 (0-0) | - | - | - | 4.9 |
| jbpi | vector | 16x15 | R8_nc128 (128x384x384) | 128 | 3 (3) | 6.58 (6.31-7.02) | 6.93 | 62 (45-71) | 1.33 | 319 | 0.01 | 22.5 |
| jbpi | scalar | 16x15 | R8_nc192 (128x384x384) | 192 | 3 (0) | 8.97 (8.90-9.05) | 9.17 | 0 (0-0) | - | - | - | 5.6 |
| jbpi | vector | 16x15 | R8_nc192 (128x384x384) | 192 | 3 (3) | 6.43 (6.25-6.73) | 6.00 | 74 (68-76) | 0.38 | 94 | 0.02 | 25.0 |
| jbpi | scalar | 16x15 | R8_nc240 (128x384x384) | 240 | 3 (1) | 8.54 (8.39-8.78) | 9.05 | 9 (0-28) | 1.07 | 258 | 0.04 | 9.8 |
| jbpi | vector | 16x15 | R8_nc240 (128x384x384) | 240 | 3 (3) | 6.89 (6.38-7.77) | 6.81 | 60 (38-74) | 0.47 | 110 | 0.02 | 24.2 |
| jbpi | scalar | 16x15 | R8_nc256 (128x384x384) | 256 | 3 (2) | 8.63 (8.56-8.67) | 8.93 | 9 (0-16) | 4.98 | 1180 | 0.01 | 7.5 |
| jbpi | vector | 16x15 | R8_nc256 (128x384x384) | 256 | 3 (3) | 6.58 (5.96-7.62) | 6.55 | 58 (34-73) | 1.51 | 373 | 0.01 | 25.1 |
| jbpi | scalar | 16x15 | R8_nc32 (128x384x384) | 32 | 3 (0) | 5.78 (5.76-5.80) | 5.78 | 0 (0-0) | - | - | - | 0.6 |
| jbpi | vector | 16x15 | R8_nc32 (128x384x384) | 32 | 3 (0) | 6.80 (6.70-6.88) | 6.86 | 0 (0-0) | - | - | - | 2.2 |
| jbpi | scalar | 16x15 | R8_nc320 (128x384x384) | 320 | 3 (2) | 8.50 (8.33-8.82) | 8.84 | 12 (0-20) | 0.92 | 216 | 0.01 | 7.5 |
| jbpi | vector | 16x15 | R8_nc320 (128x384x384) | 320 | 3 (3) | 6.06 (5.92-6.25) | 6.03 | 76 (71-81) | 0.23 | 54 | 0.02 | 24.9 |
| jbpi | scalar | 16x15 | R8_nc375 (128x384x384) | 375 | 3 (1) | 8.64 (8.38-8.90) | 9.03 | 6 (0-19) | 1.78 | 416 | 0.01 | 8.1 |
| jbpi | vector | 16x15 | R8_nc375 (128x384x384) | 375 | 3 (3) | 6.32 (5.85-7.19) | 6.30 | 63 (41-77) | 0.50 | 120 | 0.02 | 27.5 |
| jbpi | scalar | 16x15 | R8_nc384 (128x384x384) | 384 | 3 (2) | 8.76 (8.58-9.08) | 9.31 | 14 (0-29) | 1.86 | 466 | 0.02 | 10.9 |
| jbpi | vector | 16x15 | R8_nc384 (128x384x384) | 384 | 3 (3) | 6.44 (5.81-7.67) | 6.71 | 66 (32-84) | 1.09 | 281 | 0.02 | 25.9 |
| jbpi | scalar | 16x15 | R8_nc64 (128x384x384) | 64 | 3 (0) | 7.37 (7.35-7.39) | 7.37 | 0 (0-0) | - | - | - | 0.7 |
| jbpi | vector | 16x15 | R8_nc64 (128x384x384) | 64 | 3 (0) | 8.15 (8.10-8.19) | 8.24 | 2 (0-4) | - | - | - | 3.5 |
| jbpi | scalar | 16x15 | R8_nc96 (128x384x384) | 96 | 3 (0) | 8.12 (8.11-8.14) | 8.12 | 0 (0-0) | - | - | - | 1.0 |
| jbpi | vector | 16x15 | R8_nc96 (128x384x384) | 96 | 3 (3) | 7.58 (7.15-8.02) | 8.43 | 30 (13-46) | 1.16 | 268 | 0.01 | 16.9 |
| jbpi2 | vector | 32x6 | B1 (1024x384x1024) | 384 | 4 (0) | 7.97 (7.95-8.02) | 7.97 | 0 (0-0) | - | - | - | 0.3 |
| jbpi2 | vector | 32x7 | B1 (1024x384x1024) | 384 | 4 (0) | 8.12 (8.09-8.14) | 8.13 | 0 (0-0) | - | - | - | 0.2 |
| jbpi2 | vector | 16x15 | B2 (384x384x64) | 384 | 21 (0) | 12.41 (12.23-12.52) | 12.48 | 0 (0-0) | - | - | - | 2.0 |
| jbpi2 | vector+zeroline | 16x15 | B2 (384x384x64) | 384 | 21 (0) | 12.27 (11.99-12.39) | 12.37 | 0 (0-0) | - | - | - | 2.5 |
| jbpi2 | vector | 16x15 | B3 (64x384x384) | 384 | 21 (21) | 4.64 (4.37-5.21) | 4.31 | 87 (73-93) | 0.06 | 21 | 0.08 | 23.2 |
| jbpi2 | vector+zeroline | 16x15 | B3 (64x384x384) | 384 | 21 (21) | 4.47 (4.30-5.75) | 4.33 | 88 (61-93) | 0.09 | 31 | 0.05 | 20.3 |
| jbpi2 | vector | 16x15 | B3_K256 (64x384x256) | 384 | 21 (21) | 4.68 (4.54-4.86) | 4.59 | 97 (95-97) | 0.02 | 17 | 0.08 | 17.1 |
| jbpi2 | vector+zeroline | 16x15 | B3_K256 (64x384x256) | 384 | 21 (21) | 4.75 (4.53-5.29) | 4.56 | 92 (74-97) | 0.02 | 16 | 0.07 | 20.4 |
| jbpi2 | vector | 16x15 | B3_nc128 (64x384x384) | 128 | 21 (21) | 5.56 (5.38-5.97) | 5.01 | 79 (66-88) | 0.14 | 60 | 0.03 | 24.1 |
| jbpi2 | vector+zeroline | 16x15 | B3_nc128 (64x384x384) | 128 | 21 (21) | 6.01 (5.69-6.61) | 5.98 | 61 (47-70) | 0.27 | 115 | 0.02 | 25.0 |
| jbpi2 | vector | 32x6 | B4 (4096x384x1024) | 384 | 3 (0) | 7.80 (7.71-7.87) | 7.80 | 0 (0-0) | - | - | - | 0.1 |
| jbpi2 | vector | 32x7 | B4 (4096x384x1024) | 384 | 3 (0) | 8.09 (7.96-8.16) | 8.09 | 0 (0-0) | - | - | - | 0.1 |
| jbpi2 | vector | 32x6 | B5 (1024x384x4096) | 384 | 4 (0) | 7.69 (7.68-7.71) | 7.69 | 0 (0-0) | - | - | - | 0.1 |
| jbpi2 | vector | 32x7 | B5 (1024x384x4096) | 384 | 4 (0) | 7.75 (7.66-7.83) | 7.75 | 0 (0-0) | - | - | - | 0.1 |
| jbpi2 | vector | 16x15 | R24 (384x384x384) | 384 | 21 (21) | 7.96 (7.71-8.27) | 8.14 | 40 (27-56) | 1.23 | 96 | 0.03 | 12.9 |
| jbpi2 | vector+zeroline | 16x15 | R24 (384x384x384) | 384 | 21 (10) | 8.32 (7.85-8.62) | 8.64 | 13 (0-37) | 2.05 | 162 | 0.03 | 11.4 |
| jbpi2 | vector | 32x6 | S1 (1000x1000x1000) | 1000 | 3 (0) | 8.32 (8.30-8.34) | 8.32 | 0 (0-0) | - | - | - | 0.1 |
| jbpi2 | vector | 32x7 | S1 (1000x1000x1000) | 1000 | 3 (0) | 8.89 (8.89-8.91) | 8.89 | 0 (0-0) | - | - | - | 0.1 |
| jbpi2 | vector | 32x6 | S2 (2000x2000x2000) | 1024 | 3 (0) | 8.46 (8.41-8.54) | 8.46 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x7 | S2 (2000x2000x2000) | 1024 | 3 (0) | 9.20 (9.15-9.23) | 9.20 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x6 | S3 (3000x3000x3000) | 1024 | 3 (0) | 8.73 (8.71-8.74) | 8.73 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x7 | S3 (3000x3000x3000) | 1024 | 3 (0) | 9.39 (9.33-9.46) | 9.39 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x6 | S4 (4000x4000x4000) | 1024 | 3 (0) | 8.65 (8.61-8.68) | 8.65 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x7 | S4 (4000x4000x4000) | 1024 | 3 (0) | 9.22 (9.16-9.28) | 9.22 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x6 | S5 (5000x5000x5000) | 1024 | 3 (0) | 8.83 (8.80-8.86) | 8.83 | 0 (0-0) | - | - | - | 0.0 |
| jbpi2 | vector | 32x7 | S5 (5000x5000x5000) | 1024 | 3 (0) | 9.53 (9.45-9.61) | 9.53 | 0 (0-0) | - | - | - | 0.0 |

## 3. Time, reps or bytes

Bytes packed per rep = 4 * (K*N + M*K*ceil(N/nc)) (fp32).

| board | packer | tile | case | nc | launches (slow state) | reps before (min / median / max) | seconds before (min / median / max) | MiB packed before (mean) | packed MiB per rep | CV % reps / s / bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| jbpi | vector | 16x15 | B3 | 384 | 21 (21) | 3 / 14 / 80 | 0.007 / 0.036 / 0.244 | 14 | 0.66 | 85 / 94 / 85 |
| jbpi | vector+zeroline | 16x15 | B3 | 384 | 21 (21) | 7 / 12 / 55 | 0.016 / 0.034 / 0.156 | 11 | 0.66 | 68 / 76 / 68 |
| jbpi | scalar | 16x15 | B3_K256 | 384 | 3 (3) | 28 / 31 / 38 | 0.043 / 0.046 / 0.060 | 14 | 0.44 | 13 / 15 / 13 |
| jbpi | vector | 16x15 | B3_K256 | 384 | 24 (24) | 5 / 13 / 39 | 0.008 / 0.018 / 0.055 | 7 | 0.44 | 65 / 65 / 65 |
| jbpi | vector+zeroline | 16x15 | B3_K256 | 384 | 21 (21) | 6 / 14 / 27 | 0.008 / 0.020 / 0.037 | 6 | 0.44 | 39 / 40 / 39 |
| jbpi | scalar | 16x15 | B3_K256_nc128 | 128 | 3 (3) | 141 / 236 / 251 | 0.224 / 0.372 / 0.401 | 118 | 0.56 | 23 / 23 / 23 |
| jbpi | vector | 16x15 | B3_K256_nc128 | 128 | 3 (3) | 9 / 11 / 24 | 0.013 / 0.016 / 0.037 | 8 | 0.56 | 45 / 48 / 45 |
| jbpi | scalar | 16x15 | B3_N192 | 192 | 3 (2) | 1335 / 1422 / 1509 | 1.531 / 1.611 / 1.690 | 533 | 0.38 | 6 / 5 / 6 |
| jbpi | vector | 16x15 | B3_N192 | 192 | 3 (3) | 83 / 128 / 218 | 0.089 / 0.133 / 0.229 | 54 | 0.38 | 39 / 39 / 39 |
| jbpi | vector | 16x15 | B3_N192_nc128 | 128 | 3 (3) | 135 / 409 / 479 | 0.159 / 0.490 / 0.555 | 160 | 0.47 | 44 / 43 / 44 |
| jbpi | scalar | 16x15 | B3_N256 | 256 | 3 (3) | 263 / 2078 / 2616 | 0.412 / 3.183 / 4.061 | 775 | 0.47 | 61 / 61 / 61 |
| jbpi | vector | 16x15 | B3_N256 | 256 | 3 (3) | 51 / 149 / 154 | 0.078 / 0.232 / 0.234 | 55 | 0.47 | 40 / 40 / 40 |
| jbpi | vector | 16x15 | B3_N256_nc128 | 128 | 3 (3) | 76 / 144 / 152 | 0.116 / 0.225 / 0.236 | 70 | 0.56 | 27 / 28 / 27 |
| jbpi | scalar | 16x15 | B3_N320 | 320 | 3 (3) | 202 / 410 / 1000 | 0.381 / 0.779 / 1.972 | 302 | 0.56 | 63 / 65 / 63 |
| jbpi | vector | 16x15 | B3_N320 | 320 | 3 (3) | 36 / 97 / 222 | 0.067 / 0.202 / 0.394 | 67 | 0.56 | 65 / 61 / 65 |
| jbpi | scalar | 16x15 | B3_N320_nc128 | 128 | 3 (1) | 817 / 817 / 817 | 1.728 / 1.728 / 1.728 | 613 | 0.75 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | B3_N320_nc128 | 128 | 3 (3) | 81 / 102 / 220 | 0.161 / 0.199 / 0.447 | 101 | 0.75 | 46 / 47 / 46 |
| jbpi | scalar | 16x15 | B3_nc120 | 120 | 3 (2) | 1281 / 1564 / 1846 | 3.166 / 3.814 / 4.461 | 1466 | 0.94 | 18 / 17 / 18 |
| jbpi | vector | 16x15 | B3_nc120 | 120 | 3 (3) | 14 / 116 / 154 | 0.034 / 0.269 / 0.366 | 89 | 0.94 | 62 / 63 / 62 |
| jbpi | scalar | 16x15 | B3_nc128 | 128 | 3 (1) | 1915 / 1915 / 1915 | 4.690 / 4.690 / 4.690 | 1616 | 0.84 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | B3_nc128 | 128 | 24 (24) | 5 / 55 / 218 | 0.013 / 0.120 / 0.607 | 55 | 0.84 | 68 / 80 / 68 |
| jbpi | vector+zeroline | 16x15 | B3_nc128 | 128 | 21 (21) | 62 / 110 / 792 | 0.144 / 0.263 / 1.740 | 153 | 0.84 | 89 / 85 / 89 |
| jbpi | scalar | 16x15 | B3_nc192 | 192 | 3 (3) | 256 / 397 / 459 | 0.608 / 0.935 / 1.052 | 278 | 0.75 | 23 / 22 / 23 |
| jbpi | vector | 16x15 | B3_nc192 | 192 | 3 (3) | 79 / 84 / 115 | 0.170 / 0.183 / 0.249 | 70 | 0.75 | 17 / 17 / 17 |
| jbpi | scalar | 16x15 | B3_nc240 | 240 | 3 (3) | 209 / 567 / 1219 | 0.493 / 1.352 / 2.858 | 499 | 0.75 | 63 / 62 / 63 |
| jbpi | vector | 16x15 | B3_nc240 | 240 | 3 (3) | 56 / 123 / 171 | 0.134 / 0.261 / 0.383 | 88 | 0.75 | 40 / 39 / 40 |
| jbpi | scalar | 16x15 | B3_nc256 | 256 | 3 (3) | 276 / 287 / 2101 | 0.672 / 0.719 / 5.055 | 666 | 0.75 | 97 / 96 / 97 |
| jbpi | vector | 16x15 | B3_nc256 | 256 | 3 (3) | 57 / 80 / 98 | 0.132 / 0.193 / 0.230 | 59 | 0.75 | 21 / 22 / 21 |
| jbpi | scalar | 16x15 | B3_nc320 | 320 | 3 (2) | 396 / 426 / 456 | 0.949 / 1.032 / 1.115 | 320 | 0.75 | 7 / 8 / 7 |
| jbpi | vector | 16x15 | B3_nc320 | 320 | 3 (3) | 37 / 49 / 87 | 0.085 / 0.115 / 0.213 | 43 | 0.75 | 37 / 40 / 37 |
| jbpi | scalar | 16x15 | B3_nc375 | 375 | 3 (3) | 513 / 773 / 1541 | 1.203 / 1.759 / 3.588 | 707 | 0.75 | 46 / 47 / 46 |
| jbpi | vector | 16x15 | B3_nc375 | 375 | 3 (3) | 29 / 58 / 61 | 0.069 / 0.131 / 0.133 | 37 | 0.75 | 29 / 27 / 29 |
| jbpi | scalar | 16x15 | B3_nc384 | 384 | 3 (3) | 188 / 209 / 247 | 0.440 / 0.473 / 0.565 | 141 | 0.66 | 11 / 11 / 11 |
| jbpi | vector | 16x15 | B3_nc384 | 384 | 3 (3) | 44 / 50 / 97 | 0.095 / 0.108 / 0.214 | 42 | 0.66 | 37 / 38 / 37 |
| jbpi | vector | 16x15 | B3_nc64 | 64 | 3 (3) | 457 / 484 / 1357 | 1.139 / 1.202 / 3.472 | 862 | 1.12 | 55 / 56 / 55 |
| jbpi | vector | 16x15 | B3_nc96 | 96 | 3 (3) | 83 / 181 / 190 | 0.202 / 0.445 / 0.456 | 142 | 0.94 | 32 / 32 / 32 |
| jbpi | vector | 16x15 | R12_nc120 | 120 | 3 (3) | 73 / 74 / 358 | 0.448 / 0.450 / 2.148 | 284 | 1.69 | 80 / 79 / 80 |
| jbpi | vector | 16x15 | R12_nc128 | 128 | 3 (3) | 96 / 98 / 292 | 0.595 / 0.603 / 1.820 | 228 | 1.41 | 57 / 57 / 57 |
| jbpi | vector | 16x15 | R12_nc192 | 192 | 3 (3) | 38 / 74 / 77 | 0.223 / 0.435 / 0.441 | 71 | 1.12 | 28 / 28 / 28 |
| jbpi | vector | 16x15 | R12_nc240 | 240 | 3 (3) | 47 / 89 / 137 | 0.281 / 0.516 / 0.792 | 102 | 1.12 | 40 / 39 / 40 |
| jbpi | vector | 16x15 | R12_nc256 | 256 | 3 (3) | 44 / 45 / 115 | 0.264 / 0.278 / 0.713 | 76 | 1.12 | 49 / 50 / 49 |
| jbpi | vector | 16x15 | R12_nc320 | 320 | 3 (3) | 33 / 64 / 106 | 0.201 / 0.387 / 0.666 | 76 | 1.12 | 44 / 46 / 44 |
| jbpi | vector | 16x15 | R12_nc375 | 375 | 3 (3) | 50 / 64 / 167 | 0.295 / 0.383 / 1.007 | 105 | 1.12 | 56 / 56 / 56 |
| jbpi | scalar | 16x15 | R12_nc384 | 384 | 3 (2) | 693 / 730 / 767 | 4.041 / 4.292 / 4.543 | 616 | 0.84 | 5 / 6 / 5 |
| jbpi | vector | 16x15 | R12_nc384 | 384 | 3 (3) | 37 / 68 / 141 | 0.212 / 0.385 / 0.817 | 69 | 0.84 | 53 / 54 / 53 |
| jbpi | vector | 16x15 | R12_nc64 | 64 | 3 (1) | 1 / 1 / 1 | 0.009 / 0.009 / 0.009 | 2 | 2.25 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R12_nc96 | 96 | 3 (2) | 94 / 190 / 286 | 0.592 / 1.212 / 1.833 | 321 | 1.69 | 51 / 51 / 51 |
| jbpi | vector | 16x15 | R24 | 384 | 21 (21) | 20 / 78 / 411 | 0.246 / 0.986 / 4.924 | 103 | 1.12 | 86 / 82 / 86 |
| jbpi | vector+zeroline | 16x15 | R24 | 384 | 21 (9) | 54 / 133 / 424 | 0.672 / 1.696 / 5.171 | 165 | 1.12 | 72 / 70 / 72 |
| jbpi | vector | 16x15 | R24_nc128 | 128 | 3 (1) | 144 / 144 / 144 | 1.937 / 1.937 / 1.937 | 324 | 2.25 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R24_nc192 | 192 | 3 (2) | 209 / 216 / 222 | 2.655 / 2.771 / 2.888 | 364 | 1.69 | 3 / 4 / 3 |
| jbpi | vector | 16x15 | R24_nc256 | 256 | 3 (1) | 44 / 44 / 44 | 0.594 / 0.594 / 0.594 | 74 | 1.69 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R24_nc320 | 320 | 3 (2) | 114 / 188 / 261 | 1.523 / 2.510 / 3.496 | 316 | 1.69 | 39 / 39 / 39 |
| jbpi | vector | 16x15 | R24_nc375 | 375 | 3 (1) | 245 / 245 / 245 | 3.259 / 3.259 / 3.259 | 413 | 1.69 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R24_nc384 | 384 | 3 (1) | 161 / 161 / 161 | 2.088 / 2.088 / 2.088 | 181 | 1.12 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R8_nc120 | 120 | 3 (3) | 50 / 77 / 81 | 0.208 / 0.332 / 0.336 | 91 | 1.31 | 20 / 20 / 20 |
| jbpi | vector | 16x15 | R8_nc128 | 128 | 3 (3) | 43 / 64 / 851 | 0.181 / 0.291 / 3.522 | 359 | 1.12 | 118 / 116 / 118 |
| jbpi | vector | 16x15 | R8_nc192 | 192 | 3 (3) | 41 / 83 / 157 | 0.162 / 0.328 / 0.643 | 88 | 0.94 | 51 / 53 / 51 |
| jbpi | scalar | 16x15 | R8_nc240 | 240 | 3 (1) | 258 / 258 / 258 | 1.067 / 1.067 / 1.067 | 242 | 0.94 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R8_nc240 | 240 | 3 (3) | 65 / 102 / 164 | 0.275 / 0.428 / 0.693 | 103 | 0.94 | 37 / 37 / 37 |
| jbpi | scalar | 16x15 | R8_nc256 | 256 | 3 (2) | 799 / 1180 / 1562 | 3.366 / 4.980 / 6.595 | 1107 | 0.94 | 32 / 32 / 32 |
| jbpi | vector | 16x15 | R8_nc256 | 256 | 3 (3) | 20 / 117 / 981 | 0.084 / 0.472 / 3.962 | 349 | 0.94 | 116 / 116 / 116 |
| jbpi | scalar | 16x15 | R8_nc320 | 320 | 3 (2) | 190 / 216 / 242 | 0.809 / 0.925 / 1.040 | 202 | 0.94 | 12 / 12 / 12 |
| jbpi | vector | 16x15 | R8_nc320 | 320 | 3 (3) | 44 / 48 / 70 | 0.194 / 0.195 / 0.294 | 51 | 0.94 | 21 / 20 / 21 |
| jbpi | scalar | 16x15 | R8_nc375 | 375 | 3 (1) | 416 / 416 / 416 | 1.781 / 1.781 / 1.781 | 390 | 0.94 | 0 / 0 / 0 |
| jbpi | vector | 16x15 | R8_nc375 | 375 | 3 (3) | 76 / 95 / 189 | 0.308 / 0.397 / 0.809 | 112 | 0.94 | 41 / 43 / 41 |
| jbpi | scalar | 16x15 | R8_nc384 | 384 | 3 (2) | 377 / 466 / 556 | 1.499 / 1.862 / 2.224 | 350 | 0.75 | 19 / 19 / 19 |
| jbpi | vector | 16x15 | R8_nc384 | 384 | 3 (3) | 23 / 43 / 778 | 0.091 / 0.182 / 2.999 | 211 | 0.75 | 125 / 124 / 125 |
| jbpi | vector | 16x15 | R8_nc96 | 96 | 3 (3) | 1 / 1 / 803 | 0.007 / 0.008 / 3.466 | 352 | 1.31 | 141 / 141 / 141 |
| jbpi2 | vector | 16x15 | B3 | 384 | 21 (21) | 3 / 18 / 54 | 0.006 / 0.050 / 0.149 | 14 | 0.66 | 71 / 74 / 71 |
| jbpi2 | vector+zeroline | 16x15 | B3 | 384 | 21 (21) | 8 / 27 / 71 | 0.022 / 0.075 / 0.220 | 20 | 0.66 | 60 / 66 / 60 |
| jbpi2 | vector | 16x15 | B3_K256 | 384 | 21 (21) | 8 / 13 / 35 | 0.011 / 0.019 / 0.048 | 8 | 0.44 | 53 / 51 / 53 |
| jbpi2 | vector+zeroline | 16x15 | B3_K256 | 384 | 21 (21) | 4 / 15 / 34 | 0.006 / 0.021 / 0.048 | 7 | 0.44 | 48 / 49 / 48 |
| jbpi2 | vector | 16x15 | B3_nc128 | 128 | 21 (21) | 8 / 48 / 189 | 0.019 / 0.104 / 0.466 | 51 | 0.84 | 67 / 71 / 67 |
| jbpi2 | vector+zeroline | 16x15 | B3_nc128 | 128 | 21 (21) | 34 / 94 / 257 | 0.076 / 0.219 / 0.595 | 97 | 0.84 | 57 / 57 / 57 |
| jbpi2 | vector | 16x15 | R24 | 384 | 21 (21) | 18 / 80 / 221 | 0.238 / 1.043 / 2.857 | 108 | 1.12 | 52 / 52 / 52 |
| jbpi2 | vector+zeroline | 16x15 | R24 | 384 | 21 (10) | 82 / 140 / 306 | 1.030 / 1.764 / 3.790 | 182 | 1.12 | 36 / 36 / 36 |

Spread across shapes of each case's mean switch point (CV %, cases with 2 or more slow-state launches; lower = closer to constant):

| board | packer | cases | reps | seconds | bytes packed | closest to constant |
|---|---|---|---|---|---|---|
| jbpi | vector | 40 | 92 | 105 | 107 | reps |
| jbpi | vector+zeroline | 4 | 84 | 128 | 90 | reps |
| jbpi | scalar | 16 | 69 | 75 | 73 | reps |
| jbpi2 | vector | 4 | 66 | 139 | 89 | reps |
| jbpi2 | vector+zeroline | 4 | 74 | 138 | 91 | reps |

## 4. Shape features

| slow reps % | board | packer | tile | case | M | N | K | nc | mc | kc | jc x pc x ic blocks | B block KiB (>L2?) | A block KiB | packed MiB/rep | packed B per kFLOP | lda=M 2^k | ldb=K 2^k | B block 2^k | A block 2^k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 97 | jbpi2 | vector | 16x15 | B3_K256 | 64 | 384 | 256 | 384 | 64 | 128 | 1x2x1 | 192 (no) | 32 | 0.44 | 36.5 | yes | yes | no | yes |
| 95 | jbpi | vector+zeroline | 16x15 | B3_K256 | 64 | 384 | 256 | 384 | 64 | 128 | 1x2x1 | 192 (no) | 32 | 0.44 | 36.5 | yes | yes | no | yes |
| 95 | jbpi | vector | 16x15 | B3_K256 | 64 | 384 | 256 | 384 | 64 | 128 | 1x2x1 | 192 (no) | 32 | 0.44 | 36.5 | yes | yes | no | yes |
| 93 | jbpi | scalar | 16x15 | B3_K256 | 64 | 384 | 256 | 384 | 64 | 128 | 1x2x1 | 192 (no) | 32 | 0.44 | 36.5 | yes | yes | no | yes |
| 92 | jbpi2 | vector+zeroline | 16x15 | B3_K256 | 64 | 384 | 256 | 384 | 64 | 128 | 1x2x1 | 192 (no) | 32 | 0.44 | 36.5 | yes | yes | no | yes |
| 89 | jbpi | vector | 16x15 | B3_K256_nc128 | 64 | 384 | 256 | 128 | 64 | 128 | 3x2x1 | 64 (no) | 32 | 0.56 | 46.9 | yes | yes | yes | yes |
| 89 | jbpi | vector | 16x15 | B3 | 64 | 384 | 384 | 384 | 64 | 128 | 1x3x1 | 192 (no) | 32 | 0.66 | 36.5 | yes | no | no | yes |
| 88 | jbpi2 | vector+zeroline | 16x15 | B3 | 64 | 384 | 384 | 384 | 64 | 128 | 1x3x1 | 192 (no) | 32 | 0.66 | 36.5 | yes | no | no | yes |
| 88 | jbpi | vector+zeroline | 16x15 | B3 | 64 | 384 | 384 | 384 | 64 | 128 | 1x3x1 | 192 (no) | 32 | 0.66 | 36.5 | yes | no | no | yes |
| 87 | jbpi2 | vector | 16x15 | B3 | 64 | 384 | 384 | 384 | 64 | 128 | 1x3x1 | 192 (no) | 32 | 0.66 | 36.5 | yes | no | no | yes |
| 86 | jbpi | vector | 16x15 | B3_nc384 | 64 | 384 | 384 | 384 | 64 | 128 | 1x3x1 | 192 (no) | 32 | 0.66 | 36.5 | yes | no | no | yes |
| 83 | jbpi | vector | 16x15 | B3_N256 | 64 | 256 | 384 | 256 | 64 | 128 | 1x3x1 | 128 (no) | 32 | 0.47 | 39.1 | yes | no | yes | yes |
| 82 | jbpi | vector | 16x15 | B3_nc240 | 64 | 384 | 384 | 240 | 64 | 128 | 2x3x1 | 120 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 81 | jbpi | vector | 16x15 | B3_nc320 | 64 | 384 | 384 | 320 | 64 | 128 | 2x3x1 | 160 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 80 | jbpi | vector | 16x15 | B3_N192 | 64 | 192 | 384 | 192 | 64 | 128 | 1x3x1 | 96 (no) | 32 | 0.38 | 41.7 | yes | no | no | yes |
| 79 | jbpi | vector | 16x15 | B3_nc256 | 64 | 384 | 384 | 256 | 64 | 128 | 2x3x1 | 128 (no) | 32 | 0.75 | 41.7 | yes | no | yes | yes |
| 79 | jbpi2 | vector | 16x15 | B3_nc128 | 64 | 384 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.84 | 46.9 | yes | no | yes | yes |
| 77 | jbpi | vector | 16x15 | B3_nc120 | 64 | 384 | 384 | 120 | 64 | 128 | 4x3x1 | 60 (no) | 32 | 0.94 | 52.1 | yes | no | no | yes |
| 77 | jbpi | vector | 16x15 | B3_nc375 | 64 | 384 | 384 | 375 | 64 | 128 | 2x3x1 | 188 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 76 | jbpi | vector | 16x15 | B3_nc192 | 64 | 384 | 384 | 192 | 64 | 128 | 2x3x1 | 96 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 76 | jbpi | vector | 16x15 | R8_nc320 | 128 | 384 | 384 | 320 | 128 | 128 | 2x3x1 | 160 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 75 | jbpi | vector | 16x15 | B3_N320 | 64 | 320 | 384 | 320 | 64 | 128 | 1x3x1 | 160 (no) | 32 | 0.56 | 37.5 | yes | no | no | yes |
| 74 | jbpi | scalar | 16x15 | B3_K256_nc128 | 64 | 384 | 256 | 128 | 64 | 128 | 3x2x1 | 64 (no) | 32 | 0.56 | 46.9 | yes | yes | yes | yes |
| 74 | jbpi | vector | 16x15 | R8_nc192 | 128 | 384 | 384 | 192 | 128 | 128 | 2x3x1 | 96 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 73 | jbpi | vector | 16x15 | B3_nc128 | 64 | 384 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.84 | 46.9 | yes | no | yes | yes |
| 68 | jbpi | vector | 16x15 | R12_nc375 | 192 | 384 | 384 | 375 | 192 | 128 | 2x3x1 | 188 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 68 | jbpi | vector | 16x15 | B3_N256_nc128 | 64 | 256 | 384 | 128 | 64 | 128 | 2x3x1 | 64 (no) | 32 | 0.56 | 46.9 | yes | no | yes | yes |
| 66 | jbpi | vector | 16x15 | R8_nc384 | 128 | 384 | 384 | 384 | 128 | 128 | 1x3x1 | 192 (no) | 64 | 0.75 | 20.8 | yes | no | no | yes |
| 66 | jbpi | vector | 16x15 | B3_N192_nc128 | 64 | 192 | 384 | 128 | 64 | 128 | 2x3x1 | 64 (no) | 32 | 0.47 | 52.1 | yes | no | yes | yes |
| 66 | jbpi | vector | 16x15 | B3_nc96 | 64 | 384 | 384 | 96 | 64 | 128 | 4x3x1 | 48 (no) | 32 | 0.94 | 52.1 | yes | no | no | yes |
| 64 | jbpi | vector | 16x15 | R12_nc384 | 192 | 384 | 384 | 384 | 192 | 128 | 1x3x1 | 192 (no) | 96 | 0.84 | 15.6 | no | no | no | no |
| 63 | jbpi | vector | 16x15 | R8_nc375 | 128 | 384 | 384 | 375 | 128 | 128 | 2x3x1 | 188 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 62 | jbpi | vector | 16x15 | R8_nc128 | 128 | 384 | 384 | 128 | 128 | 128 | 3x3x1 | 64 (no) | 64 | 1.12 | 31.2 | yes | no | yes | yes |
| 61 | jbpi2 | vector+zeroline | 16x15 | B3_nc128 | 64 | 384 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.84 | 46.9 | yes | no | yes | yes |
| 61 | jbpi | vector | 16x15 | B3_N320_nc128 | 64 | 320 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.75 | 50.0 | yes | no | yes | yes |
| 60 | jbpi | vector | 16x15 | R8_nc240 | 128 | 384 | 384 | 240 | 128 | 128 | 2x3x1 | 120 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 60 | jbpi | vector+zeroline | 16x15 | B3_nc128 | 64 | 384 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.84 | 46.9 | yes | no | yes | yes |
| 58 | jbpi | vector | 16x15 | R8_nc256 | 128 | 384 | 384 | 256 | 128 | 128 | 2x3x1 | 128 (no) | 64 | 0.94 | 26.0 | yes | no | yes | yes |
| 57 | jbpi | vector | 16x15 | R8_nc120 | 128 | 384 | 384 | 120 | 128 | 128 | 4x3x1 | 60 (no) | 64 | 1.31 | 36.5 | yes | no | no | yes |
| 55 | jbpi | vector | 16x15 | R12_nc256 | 192 | 384 | 384 | 256 | 192 | 128 | 2x3x1 | 128 (no) | 96 | 1.12 | 20.8 | no | no | yes | no |
| 54 | jbpi | vector | 16x15 | R12_nc192 | 192 | 384 | 384 | 192 | 192 | 128 | 2x3x1 | 96 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 49 | jbpi | vector | 16x15 | R12_nc320 | 192 | 384 | 384 | 320 | 192 | 128 | 2x3x1 | 160 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 47 | jbpi | vector | 16x15 | R12_nc240 | 192 | 384 | 384 | 240 | 192 | 128 | 2x3x1 | 120 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 47 | jbpi | vector | 16x15 | R12_nc128 | 192 | 384 | 384 | 128 | 192 | 128 | 3x3x1 | 64 (no) | 96 | 1.41 | 26.0 | no | no | yes | no |
| 46 | jbpi | scalar | 16x15 | B3_nc384 | 64 | 384 | 384 | 384 | 64 | 128 | 1x3x1 | 192 (no) | 32 | 0.66 | 36.5 | yes | no | no | yes |
| 45 | jbpi | vector | 16x15 | R12_nc120 | 192 | 384 | 384 | 120 | 192 | 128 | 4x3x1 | 60 (no) | 96 | 1.69 | 31.2 | no | no | no | no |
| 40 | jbpi | scalar | 16x15 | B3_nc240 | 64 | 384 | 384 | 240 | 64 | 128 | 2x3x1 | 120 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 40 | jbpi | vector | 16x15 | R24 | 384 | 384 | 384 | 384 | 384 | 128 | 1x3x1 | 192 (no) | 192 | 1.12 | 10.4 | no | no | no | no |
| 40 | jbpi2 | vector | 16x15 | R24 | 384 | 384 | 384 | 384 | 384 | 128 | 1x3x1 | 192 (no) | 192 | 1.12 | 10.4 | no | no | no | no |
| 38 | jbpi | scalar | 16x15 | B3_nc375 | 64 | 384 | 384 | 375 | 64 | 128 | 2x3x1 | 188 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 37 | jbpi | scalar | 16x15 | B3_nc192 | 64 | 384 | 384 | 192 | 64 | 128 | 2x3x1 | 96 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 32 | jbpi | scalar | 16x15 | B3_N320 | 64 | 320 | 384 | 320 | 64 | 128 | 1x3x1 | 160 (no) | 32 | 0.56 | 37.5 | yes | no | no | yes |
| 30 | jbpi | vector | 16x15 | R8_nc96 | 128 | 384 | 384 | 96 | 128 | 128 | 4x3x1 | 48 (no) | 64 | 1.31 | 36.5 | yes | no | no | yes |
| 29 | jbpi | scalar | 16x15 | B3_nc256 | 64 | 384 | 384 | 256 | 64 | 128 | 2x3x1 | 128 (no) | 32 | 0.75 | 41.7 | yes | no | yes | yes |
| 23 | jbpi | scalar | 16x15 | B3_nc320 | 64 | 384 | 384 | 320 | 64 | 128 | 2x3x1 | 160 (no) | 32 | 0.75 | 41.7 | yes | no | no | yes |
| 22 | jbpi | vector | 16x15 | B3_nc64 | 64 | 384 | 384 | 64 | 64 | 128 | 6x3x1 | 32 (no) | 32 | 1.12 | 62.5 | yes | no | yes | yes |
| 17 | jbpi | scalar | 16x15 | B3_N192 | 64 | 192 | 384 | 192 | 64 | 128 | 1x3x1 | 96 (no) | 32 | 0.38 | 41.7 | yes | no | no | yes |
| 16 | jbpi | scalar | 16x15 | B3_N256 | 64 | 256 | 384 | 256 | 64 | 128 | 1x3x1 | 128 (no) | 32 | 0.47 | 39.1 | yes | no | yes | yes |
| 14 | jbpi | scalar | 16x15 | R8_nc384 | 128 | 384 | 384 | 384 | 128 | 128 | 1x3x1 | 192 (no) | 64 | 0.75 | 20.8 | yes | no | no | yes |
| 14 | jbpi | vector | 16x15 | R12_nc96 | 192 | 384 | 384 | 96 | 192 | 128 | 4x3x1 | 48 (no) | 96 | 1.69 | 31.2 | no | no | no | no |
| 13 | jbpi2 | vector+zeroline | 16x15 | R24 | 384 | 384 | 384 | 384 | 384 | 128 | 1x3x1 | 192 (no) | 192 | 1.12 | 10.4 | no | no | no | no |
| 13 | jbpi | vector | 16x15 | R24_nc192 | 384 | 384 | 384 | 192 | 384 | 128 | 2x3x1 | 96 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 12 | jbpi | vector+zeroline | 16x15 | R24 | 384 | 384 | 384 | 384 | 384 | 128 | 1x3x1 | 192 (no) | 192 | 1.12 | 10.4 | no | no | no | no |
| 12 | jbpi | scalar | 16x15 | R8_nc320 | 128 | 384 | 384 | 320 | 128 | 128 | 2x3x1 | 160 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 12 | jbpi | vector | 16x15 | R24_nc320 | 384 | 384 | 384 | 320 | 384 | 128 | 2x3x1 | 160 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 10 | jbpi | scalar | 16x15 | B3_nc120 | 64 | 384 | 384 | 120 | 64 | 128 | 4x3x1 | 60 (no) | 32 | 0.94 | 52.1 | yes | no | no | yes |
| 9 | jbpi | scalar | 16x15 | R8_nc256 | 128 | 384 | 384 | 256 | 128 | 128 | 2x3x1 | 128 (no) | 64 | 0.94 | 26.0 | yes | no | yes | yes |
| 9 | jbpi | scalar | 16x15 | R8_nc240 | 128 | 384 | 384 | 240 | 128 | 128 | 2x3x1 | 120 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 9 | jbpi | vector | 16x15 | R24_nc128 | 384 | 384 | 384 | 128 | 384 | 128 | 3x3x1 | 64 (no) | 192 | 2.25 | 20.8 | no | no | yes | no |
| 8 | jbpi | vector | 16x15 | R24_nc375 | 384 | 384 | 384 | 375 | 384 | 128 | 2x3x1 | 188 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 7 | jbpi | vector | 16x15 | R24_nc256 | 384 | 384 | 384 | 256 | 384 | 128 | 2x3x1 | 128 (no) | 192 | 1.69 | 15.6 | no | no | yes | no |
| 7 | jbpi | scalar | 16x15 | R12_nc384 | 192 | 384 | 384 | 384 | 192 | 128 | 1x3x1 | 192 (no) | 96 | 0.84 | 15.6 | no | no | no | no |
| 6 | jbpi | scalar | 16x15 | R8_nc375 | 128 | 384 | 384 | 375 | 128 | 128 | 2x3x1 | 188 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 6 | jbpi | vector | 16x15 | R24_nc384 | 384 | 384 | 384 | 384 | 384 | 128 | 1x3x1 | 192 (no) | 192 | 1.12 | 10.4 | no | no | no | no |
| 6 | jbpi | scalar | 16x15 | B3_nc128 | 64 | 384 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.84 | 46.9 | yes | no | yes | yes |
| 3 | jbpi | scalar | 16x15 | B3_N320_nc128 | 64 | 320 | 384 | 128 | 64 | 128 | 3x3x1 | 64 (no) | 32 | 0.75 | 50.0 | yes | no | yes | yes |
| 3 | jbpi | vector | 16x15 | R12_nc64 | 192 | 384 | 384 | 64 | 192 | 128 | 6x3x1 | 32 (no) | 96 | 2.25 | 41.7 | no | no | yes | no |
| 2 | jbpi | vector | 16x15 | R8_nc64 | 128 | 384 | 384 | 64 | 128 | 128 | 6x3x1 | 32 (no) | 64 | 1.69 | 46.9 | yes | no | yes | yes |
| 2 | jbpi | scalar | 16x15 | R12_nc320 | 192 | 384 | 384 | 320 | 192 | 128 | 2x3x1 | 160 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 1 | jbpi | scalar | 16x15 | R12_nc240 | 192 | 384 | 384 | 240 | 192 | 128 | 2x3x1 | 120 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 0 | jbpi | vector | 16x15 | R12_nc32 | 192 | 384 | 384 | 32 | 192 | 128 | 12x3x1 | 16 (no) | 96 | 3.94 | 72.9 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | B3_nc32 | 64 | 384 | 384 | 32 | 64 | 128 | 12x3x1 | 16 (no) | 32 | 1.69 | 93.8 | yes | no | yes | yes |
| 0 | jbpi | vector | 16x15 | B2 | 384 | 384 | 64 | 384 | 384 | 64 | 1x1x1 | 96 (no) | 96 | 0.19 | 10.4 | no | yes | no | no |
| 0 | jbpi | vector+zeroline | 16x15 | B2 | 384 | 384 | 64 | 384 | 384 | 64 | 1x1x1 | 96 (no) | 96 | 0.19 | 10.4 | no | yes | no | no |
| 0 | jbpi | scalar | 16x15 | B2_nc128 | 384 | 384 | 64 | 128 | 384 | 64 | 3x1x1 | 32 (no) | 96 | 0.38 | 20.8 | no | yes | yes | no |
| 0 | jbpi | vector | 16x15 | B2_nc128 | 384 | 384 | 64 | 128 | 384 | 64 | 3x1x1 | 32 (no) | 96 | 0.38 | 20.8 | no | yes | yes | no |
| 0 | jbpi | scalar | 16x15 | B2_nc384 | 384 | 384 | 64 | 384 | 384 | 64 | 1x1x1 | 96 (no) | 96 | 0.19 | 10.4 | no | yes | no | no |
| 0 | jbpi | vector | 16x15 | B2_nc384 | 384 | 384 | 64 | 384 | 384 | 64 | 1x1x1 | 96 (no) | 96 | 0.19 | 10.4 | no | yes | no | no |
| 0 | jbpi | scalar | 16x15 | B3_N192_nc128 | 64 | 192 | 384 | 128 | 64 | 128 | 2x3x1 | 64 (no) | 32 | 0.47 | 52.1 | yes | no | yes | yes |
| 0 | jbpi | scalar | 16x15 | B3_N256_nc128 | 64 | 256 | 384 | 128 | 64 | 128 | 2x3x1 | 64 (no) | 32 | 0.56 | 46.9 | yes | no | yes | yes |
| 0 | jbpi | vector | 16x15 | B3_nc32 | 64 | 384 | 384 | 32 | 64 | 128 | 12x3x1 | 16 (no) | 32 | 1.69 | 93.8 | yes | no | yes | yes |
| 0 | jbpi | scalar | 16x15 | B3_nc64 | 64 | 384 | 384 | 64 | 64 | 128 | 6x3x1 | 32 (no) | 32 | 1.12 | 62.5 | yes | no | yes | yes |
| 0 | jbpi | scalar | 16x15 | B3_nc96 | 64 | 384 | 384 | 96 | 64 | 128 | 4x3x1 | 48 (no) | 32 | 0.94 | 52.1 | yes | no | no | yes |
| 0 | jbpi | scalar | 16x15 | R12_nc120 | 192 | 384 | 384 | 120 | 192 | 128 | 4x3x1 | 60 (no) | 96 | 1.69 | 31.2 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R12_nc128 | 192 | 384 | 384 | 128 | 192 | 128 | 3x3x1 | 64 (no) | 96 | 1.41 | 26.0 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R12_nc192 | 192 | 384 | 384 | 192 | 192 | 128 | 2x3x1 | 96 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R12_nc256 | 192 | 384 | 384 | 256 | 192 | 128 | 2x3x1 | 128 (no) | 96 | 1.12 | 20.8 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R12_nc32 | 192 | 384 | 384 | 32 | 192 | 128 | 12x3x1 | 16 (no) | 96 | 3.94 | 72.9 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R12_nc375 | 192 | 384 | 384 | 375 | 192 | 128 | 2x3x1 | 188 (no) | 96 | 1.12 | 20.8 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R12_nc64 | 192 | 384 | 384 | 64 | 192 | 128 | 6x3x1 | 32 (no) | 96 | 2.25 | 41.7 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R12_nc96 | 192 | 384 | 384 | 96 | 192 | 128 | 4x3x1 | 48 (no) | 96 | 1.69 | 31.2 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc120 | 384 | 384 | 384 | 120 | 384 | 128 | 4x3x1 | 60 (no) | 192 | 2.81 | 26.0 | no | no | no | no |
| 0 | jbpi | vector | 16x15 | R24_nc120 | 384 | 384 | 384 | 120 | 384 | 128 | 4x3x1 | 60 (no) | 192 | 2.81 | 26.0 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc128 | 384 | 384 | 384 | 128 | 384 | 128 | 3x3x1 | 64 (no) | 192 | 2.25 | 20.8 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R24_nc192 | 384 | 384 | 384 | 192 | 384 | 128 | 2x3x1 | 96 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc240 | 384 | 384 | 384 | 240 | 384 | 128 | 2x3x1 | 120 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 0 | jbpi | vector | 16x15 | R24_nc240 | 384 | 384 | 384 | 240 | 384 | 128 | 2x3x1 | 120 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc256 | 384 | 384 | 384 | 256 | 384 | 128 | 2x3x1 | 128 (no) | 192 | 1.69 | 15.6 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R24_nc32 | 384 | 384 | 384 | 32 | 384 | 128 | 12x3x1 | 16 (no) | 192 | 7.31 | 67.7 | no | no | yes | no |
| 0 | jbpi | vector | 16x15 | R24_nc32 | 384 | 384 | 384 | 32 | 384 | 128 | 12x3x1 | 16 (no) | 192 | 7.31 | 67.7 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R24_nc320 | 384 | 384 | 384 | 320 | 384 | 128 | 2x3x1 | 160 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc375 | 384 | 384 | 384 | 375 | 384 | 128 | 2x3x1 | 188 (no) | 192 | 1.69 | 15.6 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc384 | 384 | 384 | 384 | 384 | 384 | 128 | 1x3x1 | 192 (no) | 192 | 1.12 | 10.4 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R24_nc64 | 384 | 384 | 384 | 64 | 384 | 128 | 6x3x1 | 32 (no) | 192 | 3.94 | 36.5 | no | no | yes | no |
| 0 | jbpi | vector | 16x15 | R24_nc64 | 384 | 384 | 384 | 64 | 384 | 128 | 6x3x1 | 32 (no) | 192 | 3.94 | 36.5 | no | no | yes | no |
| 0 | jbpi | scalar | 16x15 | R24_nc96 | 384 | 384 | 384 | 96 | 384 | 128 | 4x3x1 | 48 (no) | 192 | 2.81 | 26.0 | no | no | no | no |
| 0 | jbpi | vector | 16x15 | R24_nc96 | 384 | 384 | 384 | 96 | 384 | 128 | 4x3x1 | 48 (no) | 192 | 2.81 | 26.0 | no | no | no | no |
| 0 | jbpi | scalar | 16x15 | R8_nc120 | 128 | 384 | 384 | 120 | 128 | 128 | 4x3x1 | 60 (no) | 64 | 1.31 | 36.5 | yes | no | no | yes |
| 0 | jbpi | scalar | 16x15 | R8_nc128 | 128 | 384 | 384 | 128 | 128 | 128 | 3x3x1 | 64 (no) | 64 | 1.12 | 31.2 | yes | no | yes | yes |
| 0 | jbpi | scalar | 16x15 | R8_nc192 | 128 | 384 | 384 | 192 | 128 | 128 | 2x3x1 | 96 (no) | 64 | 0.94 | 26.0 | yes | no | no | yes |
| 0 | jbpi | scalar | 16x15 | R8_nc32 | 128 | 384 | 384 | 32 | 128 | 128 | 12x3x1 | 16 (no) | 64 | 2.81 | 78.1 | yes | no | yes | yes |
| 0 | jbpi | vector | 16x15 | R8_nc32 | 128 | 384 | 384 | 32 | 128 | 128 | 12x3x1 | 16 (no) | 64 | 2.81 | 78.1 | yes | no | yes | yes |
| 0 | jbpi | scalar | 16x15 | R8_nc64 | 128 | 384 | 384 | 64 | 128 | 128 | 6x3x1 | 32 (no) | 64 | 1.69 | 46.9 | yes | no | yes | yes |
| 0 | jbpi | scalar | 16x15 | R8_nc96 | 128 | 384 | 384 | 96 | 128 | 128 | 4x3x1 | 48 (no) | 64 | 1.31 | 36.5 | yes | no | no | yes |
| 0 | jbpi2 | vector | 32x6 | B1 | 1024 | 384 | 1024 | 384 | 768 | 128 | 1x8x2 | 192 (no) | 384 | 5.50 | 7.2 | yes | yes | no | no |
| 0 | jbpi2 | vector | 32x7 | B1 | 1024 | 384 | 1024 | 384 | 768 | 128 | 1x8x2 | 192 (no) | 384 | 5.50 | 7.2 | yes | yes | no | no |
| 0 | jbpi2 | vector | 16x15 | B2 | 384 | 384 | 64 | 384 | 384 | 64 | 1x1x1 | 96 (no) | 96 | 0.19 | 10.4 | no | yes | no | no |
| 0 | jbpi2 | vector+zeroline | 16x15 | B2 | 384 | 384 | 64 | 384 | 384 | 64 | 1x1x1 | 96 (no) | 96 | 0.19 | 10.4 | no | yes | no | no |
| 0 | jbpi2 | vector | 32x6 | B4 | 4096 | 384 | 1024 | 384 | 768 | 128 | 1x8x6 | 192 (no) | 384 | 17.50 | 5.7 | yes | yes | no | no |
| 0 | jbpi2 | vector | 32x7 | B4 | 4096 | 384 | 1024 | 384 | 768 | 128 | 1x8x6 | 192 (no) | 384 | 17.50 | 5.7 | yes | yes | no | no |
| 0 | jbpi2 | vector | 32x6 | B5 | 1024 | 384 | 4096 | 384 | 768 | 128 | 1x32x2 | 192 (no) | 384 | 22.00 | 7.2 | yes | yes | no | no |
| 0 | jbpi2 | vector | 32x7 | B5 | 1024 | 384 | 4096 | 384 | 768 | 128 | 1x32x2 | 192 (no) | 384 | 22.00 | 7.2 | yes | yes | no | no |
| 0 | jbpi2 | vector | 32x6 | S1 | 1000 | 1000 | 1000 | 1000 | 768 | 128 | 1x8x2 | 500 (no) | 384 | 7.63 | 4.0 | no | no | no | no |
| 0 | jbpi2 | vector | 32x7 | S1 | 1000 | 1000 | 1000 | 1000 | 768 | 128 | 1x8x2 | 500 (no) | 384 | 7.63 | 4.0 | no | no | no | no |
| 0 | jbpi2 | vector | 32x6 | S2 | 2000 | 2000 | 2000 | 1024 | 768 | 128 | 2x16x3 | 512 (no) | 384 | 45.78 | 3.0 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x7 | S2 | 2000 | 2000 | 2000 | 1024 | 768 | 128 | 2x16x3 | 512 (no) | 384 | 45.78 | 3.0 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x6 | S3 | 3000 | 3000 | 3000 | 1024 | 768 | 128 | 3x24x4 | 512 (no) | 384 | 137.33 | 2.7 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x7 | S3 | 3000 | 3000 | 3000 | 1024 | 768 | 128 | 3x24x4 | 512 (no) | 384 | 137.33 | 2.7 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x6 | S4 | 4000 | 4000 | 4000 | 1024 | 768 | 128 | 4x32x6 | 512 (no) | 384 | 305.18 | 2.5 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x7 | S4 | 4000 | 4000 | 4000 | 1024 | 768 | 128 | 4x32x6 | 512 (no) | 384 | 305.18 | 2.5 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x6 | S5 | 5000 | 5000 | 5000 | 1024 | 768 | 128 | 5x40x7 | 512 (no) | 384 | 572.20 | 2.4 | no | no | yes | no |
| 0 | jbpi2 | vector | 32x7 | S5 | 5000 | 5000 | 5000 | 1024 | 768 | 128 | 5x40x7 | 512 (no) | 384 | 572.20 | 2.4 | no | no | yes | no |
