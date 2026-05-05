import math

# Hardware Configurations Registry
# Parameters: NL (sets), CL (line size), WL (associativity)
HARDWARE_CONFIGS = {
    "k1": {
        "name": "SpacemiT K1 / C910",
        "L1": {"nl": 128,  "cl": 64, "wl": 4},
        "L2": {"nl": 1024, "cl": 64, "wl": 8},
        "L3": {"nl": 4096, "cl": 64, "wl": 16} # Placeholder or cluster-level cache
    },
    "bananapi": {
        "name": "Bananapi",
        "L1": {"nl": 128,  "cl": 64, "wl": 4},
        "L2": {"nl": 1024, "cl": 64, "wl": 8},
        "L3": {"nl": 4096, "cl": 64, "wl": 16}
    }
}

def model_level(is_l3, nl, cl, wl, data_size, m, n):
    """
    Python implementation of the analytical model from Low et al. (2016).
    """
    if wl == 2:
        if not is_l3:
            k = (nl * cl) / (2.0 * m * data_size)
        else:
            k = (nl * cl) / (2.0 * n * data_size)
    else:
        div = (1.0 + float(n) / float(m))
        # Reference formula (Low et al. 2016): subtract 1 from WL for the non-resident operand
        # Both L1/L2 and L3 use (WL-1) — matches model_level.c line 38
        c_ar = math.floor((float(wl) - 1.0) / div)

        if c_ar == 0:
            c_ar = 1.0
            c_br = float(wl) - 2.0
            if not is_l3:
                k = (c_br * nl * cl) / (n * data_size)
            else:
                k = (c_br * nl * cl) / (m * data_size)
        else:
            c_br = math.ceil((float(n) / float(m)) * float(c_ar))
            if not is_l3:
                k = (c_ar * nl * cl) / (m * data_size)
            else:
                # L3 specifically: nc = c_br * NL3 * CL3 / (n * data_size)
                k = (c_br * nl * cl) / (n * data_size)
    
    return int(math.floor(k))

def get_optim_mc_nc_kc(arch_key, data_size, m_max, n_max, k_max, mr, nr):
    """
    Computes optimal BLIS parameters based on architecture and micro-tile dimensions.
    """
    if arch_key not in HARDWARE_CONFIGS:
        raise ValueError(f"Unknown architecture: {arch_key}")
    
    conf = HARDWARE_CONFIGS[arch_key]
    
    # KC: Constrained by L1
    kc = model_level(False, conf["L1"]["nl"], conf["L1"]["cl"], conf["L1"]["wl"], data_size, mr, nr)
    kc = min(k_max, kc)
    
    # MC: Constrained by L2
    mc = model_level(False, conf["L2"]["nl"], conf["L2"]["cl"], conf["L2"]["wl"], data_size, kc, nr)
    mc = min(m_max, mc)
    
    # NC: Constrained by L3 (or just a large multiple if L3 is missing)
    if conf["L3"]["nl"] > 0:
        nc = model_level(True, conf["L3"]["nl"], conf["L3"]["cl"], conf["L3"]["wl"], data_size, kc, mc)
        nc = min(n_max, nc)
    else:
        nc = 1024 # Default fallback
        
    return mc, nc, kc
