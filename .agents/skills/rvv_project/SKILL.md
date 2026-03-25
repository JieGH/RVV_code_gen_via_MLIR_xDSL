---
name: rvv_project_conventions
description: Project-specific conventions for the RVV xDSL microkernel codebase. Covers Python environment, command execution, reporting, and workflow practices.
---

# RVV xDSL Project Conventions

## 1. Python Environment

**Always use `.venv` for this project.** The virtual environment contains `xdsl` and `xdsltemplate`. Do NOT use the system Python.

```bash
# Correct pattern:
PYTHONPATH=src .venv/bin/python <script>
PYTHONPATH=src .venv/bin/python -c "..."
```

---

## 2. Git Commits — Always Use `--no-verify`

This project has pre-commit hooks (formatting, linting) that may fail on work-in-progress code. **Always add `--no-verify` when committing:**

```bash
git add -A
git commit --no-verify -m "your message"
git push
```

> **IMPORTANT**: `--no-verify` is only for this specific project (`RVV_code_gen_via_MLIR_xDSL`). Do NOT use it by default in other projects.

---

## 2. Avoiding Command Execution Stalls

Python commands that import `xdsl` or `xdsltemplate` can take **30–120 seconds** to start due to heavy module loading. **Never wait inline for output.** Instead:

### Rule: Always redirect output to a file and read it separately

```bash
# Step 1: Launch in background with redirect
PYTHONPATH=src .venv/bin/python my_script.py > /tmp/out.txt 2>&1 &

# Step 2: Poll until file has content
sleep 5 && cat /tmp/out.txt

# Or for inline one-liners:
PYTHONPATH=src .venv/bin/python -c "..." > /tmp/out.txt 2>&1 &
# Wait ~30s then:
cat /tmp/out.txt
```

The same applies to `mlir-translate` commands — the LLVM build binary at:
```
/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate
```
can take 1–2 minutes to execute.

---

## 3. mlir-translate Usage

When calling `mlir-translate` for EmitC → C++ translation:
- Use the **LLVM build binary** (NOT `/usr/local/bin/mlir-translate` which is outdated)
- Use **`print_generic_format=True`** when writing the `.mlir` file
- Pass **`-allow-unregistered-dialect`**
- Write output to a **file** with `-o`

```bash
/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate \
    -allow-unregistered-dialect -mlir-to-cpp \
    input.mlir -o output.c
```

---

## 4. Report Before Complex Implementation

When the user requests changes that involve **more than 2 files** or **new dialect ops + lowerings + test scripts**, always:

1. Write an `implementation_plan.md` artifact
2. Present it to the user with `notify_user` and `BlockedOnUser=true`
3. Wait for explicit approval before writing code

Do NOT skip straight to implementation for anything involving:
- New dialect types or ops in `rvv.py`
- Changes to `rvv_to_emitc.py` lowering passes
- A new test file alongside dialect changes

---

## 5. Key File Locations

| Role | Path |
|---|---|
| RVV dialect types/ops | `src/xdsltemplate/dialects/rvv.py` |
| EmitC lowering patterns | `src/xdsltemplate/transforms/rvv_to_emitc.py` |
| Main GEMM test | `tests/test_codeGeneration.py` |
| Mixed precision test | `tests/test_mixed_precision_arith.py` |
| LLVM mlir-translate | `/home/jlei/Documents/ws_MLIR/llvm_pj_dir_06092025/llvm-project/build/bin/mlir-translate` |
