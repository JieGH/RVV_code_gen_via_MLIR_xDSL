# xDSL RISC-V Microkernel Template

This repository provides an automated infrastructure to generate, deploy, and benchmark xDSL microkernels targeting RISC-V Vector (RVV) architectures. 

### How This Project Works
At a high level, the project workflow bridges two machines:
1. **Local x86 Machine**: Responsible for code generation. The Python-based `xdsl` compiler generates the necessary GEMM C++ microkernels locally.
2. **Remote RISC-V Board**: The generated artifacts, benchmarking tools, and datasets are automatically transferred (via `rsync`) to the RISC-V environment, where all compilation and performance testing takes place. 

Once execution finishes on the RISC-V board, the benchmarking results are automatically fetched back and **stored on your local x86 machine** for analysis.

## 📊 Pipeline Overview

The core of this repository is `compile.sh`, a unified driver script that automates the entire lifecycle:

1. **Local Code Generation**: Invokes xDSL to generate GEMM C++ microkernels for all specified kernel sizes.
2. **Environment Synchronization**: Uses `rsync` to sync locally generated kernels, C++ APIs, tests, and dataset files to the RISC-V device.
3. **Unit Testing**: Evaluates the generated microkernel logic locally and runs a benchmarking C++ wrapper on the board to ensure correctness.
4. **XDSL Kernel Sweep Benchmark**: Builds the benchmark driver on the board, performs dynamic shape/mode parameter sweeps via target datasets using caching and BLIS loop wrapping frameworks. Evaluates configurations and logs outputs.
5. **OpenBLAS Reference Comparison**: Generates reference baselines by directly executing OpenBLAS instances matching the target dataset specifications. The script securely downloads both baseline outputs, matches them, and finalizes performance scatter and bar plots comparing the metrics.

## 📈 Results and Analysis
Although intermediate object files and source codes are stored on the RISC-V machine temporarily, all final evaluation outputs are securely transferred back to your local x86 machine.

Check the `tests/output/` directory for:
- **`.txt` Logs**: Raw numeric performance logs and execution timespans extracted from the board.
- **`.csv` Files**: Parsed xDSL sweep configuration tables containing GFLOPS execution speeds.
- **`.pdf` / `.png` Plots**: Graphical scatter and bar plots visually demonstrating your xDSL microkernels' performance compared against baseline architectures like OpenBLAS.

## 🚀 Getting Started

If you want to run this project, please refer to the **[Getting Started Guide](GETTING_STARTED.md)** for detailed instructions on setting up your environment, configuring your connection to the RISC-V board, and executing the benchmark suite.
