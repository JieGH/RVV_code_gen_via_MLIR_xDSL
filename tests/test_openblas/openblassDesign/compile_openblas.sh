#!/bin/bash
# compile_openblas.sh — Compile and run OpenBLAS SGEMM benchmark
# Accepts: --dir <openblas_install_dir>   (default: $HOME/.local)
#          --zvl | --zvl256b              (flag: use ZVL256B variant name)
#          <dataset.dat>                 (passed to the benchmark binary)

OPENBLAS_DIR="${HOME}/.local"
OPENBLAS_NAME="Generic OpenBLAS"

PRECISION="fp32"
PREC_FLAG="-DFP32"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir)          OPENBLAS_DIR="$2"; shift ;;
        --zvl|--zvl256b) OPENBLAS_NAME="ZVL256B OpenBLAS" ;;
        --precision)    
            PRECISION="$2"
            if [ "$PRECISION" == "i8i8i32" ]; then
                PREC_FLAG="-DI8I8I32"
            else
                PREC_FLAG="-DFP32"
            fi
            shift ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1" ;;
    esac
    shift
done

set -- $EXTRA_ARGS

echo "=========================================================="
echo " OpenBLAS Target : $OPENBLAS_NAME"
echo " Linker Path     : $OPENBLAS_DIR"
echo "=========================================================="
echo "Compiling OpenBLAS GEMM Test..."

# Assuming standard GCC or Clang
# -O3 for optimization
# -I to specify the include directory where cblas.h is
# -L to specify the library directory where libopenblas.a / libopenblas.so is
# -lopenblas to link the OpenBLAS library

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

gcc -O3 $PREC_FLAG "${SCRIPT_DIR}/openblas_gemm_test.c" -o "${SCRIPT_DIR}/openblas_gemm_test" \
    -I${OPENBLAS_DIR}/include \
    -L${OPENBLAS_DIR}/lib \
    -lopenblas -lm

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    export LD_LIBRARY_PATH="${OPENBLAS_DIR}/lib:$LD_LIBRARY_PATH"
    "${SCRIPT_DIR}/openblas_gemm_test" "$@"
else
    echo "Compilation failed. Please check your OpenBLAS directory and linker flags."
fi
