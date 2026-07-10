import math
from typing import Dict, Tuple

# Pre-defined architecture cache configurations
# Extracted from .ext/RVV_ukernels_benchmark/gemm_blis_family/modelLevel/archs/
ARCH_CONFIGS = {
    "bananapi": {
        # Cache Level 1
        "NL1": 128,
        "CL1": 64,
        "WL1": 4,
        # Cache Level 2
        "NL2": 1024,
        "CL2": 64,
        "WL2": 8,
        # Cache Level 3
        "NL3": 4096,
        "CL3": 64,
        "WL3": 16,
    },
    "jbpi2": {
        # Cache Level 1
        "NL1": 128,
        "CL1": 64,
        "WL1": 4,
        # Cache Level 2
        "NL2": 1024,
        "CL2": 64,
        "WL2": 8,
        # Cache Level 3
        "NL3": 4096,
        "CL3": 64,
        "WL3": 16,
    },
    "amd_zen": {
        "NL1": 64, "CL1": 64, "WL1": 8,
        "NL2": 1024, "CL2": 64, "WL2": 8,
        "NL3": 32768, "CL3": 64, "WL3": 16,
    },
    "arm_carmel": {
        "NL1": 256, "CL1": 64, "WL1": 4,
        "NL2": 4096, "CL2": 64, "WL2": 8,
        "NL3": 32768, "CL3": 64, "WL3": 16,
    }
}

def model_level(isL3: bool, NL: int, CL: int, WL: int, dataSize: int, m: float, n: float) -> float:
    """
    Estimate the dimension of the panels that will fit into a given level of the cache hierarchy.
    Ported from model_level.c
    """
    if WL == 2:
        if not isL3:
            k = (NL * CL) / (2.0 * m * dataSize)
        else:
            k = (NL * CL) / (2.0 * n * dataSize)
    else:
        CAr = math.floor((WL - 1.0) / (1.0 + n / m))
        if CAr == 0:
            CAr = 1.0
            CBr = WL - 2
            if not isL3:
                k = (CBr * NL * CL) / (n * dataSize)
            else:
                k = (CBr * NL * CL) / (m * dataSize)
        else:
            CBr = math.ceil((n / m) * CAr)
            if not isL3:
                k = (CAr * NL * CL) / (m * dataSize)
            else:
                k = (CBr * NL * CL) / (n * dataSize)
    return k

def get_optim_mc_nc_kc(arch: str, dataSize: int, mr: int, nr: int, mmax: int = 100000, nmax: int = 100000, kmax: int = 100000) -> Tuple[int, int, int]:
    """
    Calculates the optimal MC, NC, KC for a given architecture and micro-kernel size.
    mmax, nmax, kmax define the absolute maximum bounds (default is arbitrarily large to just test cache capacity).
    """
    if arch not in ARCH_CONFIGS:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(ARCH_CONFIGS.keys())}")
        
    cfg = ARCH_CONFIGS[arch]
    
    # KC
    kc_f = model_level(False, cfg["NL1"], cfg["CL1"], cfg["WL1"], dataSize, mr, nr)
    kc = math.floor(kc_f)
    kc = min(kmax, kc)
    
    # MC
    mc_f = model_level(False, cfg["NL2"], cfg["CL2"], cfg["WL2"], dataSize, kc, nr)
    mc = math.floor(mc_f)
    mc = min(mmax, mc)
    
    # NC
    nc_f = model_level(True, cfg["NL3"], cfg["CL3"], cfg["WL3"], dataSize, kc, mc)
    nc = math.floor(nc_f)
    nc = min(nmax, nc)
    
    return mc, nc, kc

if __name__ == "__main__":
    # Test just like model_test.c
    dataSize = 4 # fp32
    mr, nr = 8, 8
    
    mc, nc, kc = get_optim_mc_nc_kc("bananapi", dataSize, mr, nr)
    print("-------------------------------------------------")
    print(f"[*] ARCH     : bananapi")
    print(f"[*] MR       : {mr}")
    print(f"[*] NR       : {nr}")
    print("-------------------------------------------------")
    print(f"Optimized Cache Blocking Parameters:")
    print(f"MC: {mc}, NC: {nc}, KC: {kc}")
    print("-------------------------------------------------")
