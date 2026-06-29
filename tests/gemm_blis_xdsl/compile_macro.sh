#!/bin/bash
# compile_macro.sh - Unified Macro-Kernel Generation Script

# 1. Locate the virtual environment
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

# 2. Setup PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/tests/gemm_blis_xdsl"

# 3. Define paths
GEN_SCRIPT="$PROJECT_ROOT/tests/gemm_blis_xdsl/generate_macro.py"
MLIR_TRANSLATE="$PROJECT_ROOT/../llvm-project/build/bin/mlir-translate"
OUTPUT_C="$PROJECT_ROOT/tests/gemm_blis_xdsl/gen/gemm_macro_v.c"

# 4. Run the generator
MODEL_ARG=""
if [ ! -z "$1" ]; then
    MODEL_ARG="--model $1"
fi

echo "Generating GEMM Macro-Kernel with Analytical Model (K1)..."
python3 "$GEN_SCRIPT" --arch k1 $MODEL_ARG --MR 8 --NR 8 --mlir-translate "$MLIR_TRANSLATE" --output "$OUTPUT_C"

if [ $? -eq 0 ]; then
    echo "✅ Success: $OUTPUT_C generated."
else
    echo "❌ Error: Generation failed."
    exit 1
fi
