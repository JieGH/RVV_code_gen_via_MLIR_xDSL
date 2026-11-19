cd ~/issue112_audit_run
echo "=== who/uptime"; who; uptime; ps -eo user,pcpu,comm --sort=-pcpu | head
F="-O3 -march=rv64gcv_zfh_zvfh -mabi=lp64d -I."
echo "=== gcc-14 fp32"; gcc-14 $F audit.c xdsl_ukernels_fp32.c -o a14_fp32 -lm && taskset -c 4 ./a14_fp32
echo "=== gcc-14 f16"; gcc-14 $F -DF16 audit.c xdsl_ukernels_f16f16f32.c -o a14_f16 -lm && taskset -c 4 ./a14_f16
echo "=== gcc-13 fp32"; gcc $F audit.c xdsl_ukernels_fp32.c -o a13_fp32 -lm && taskset -c 4 ./a13_fp32
echo "=== gcc-13 f16 compile"; gcc $F -DF16 audit.c xdsl_ukernels_f16f16f32.c -o a13_f16 -lm 2>&1 | head -8; echo "rc=$?"
