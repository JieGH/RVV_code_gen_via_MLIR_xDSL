"""
Issue #34 §3.3: B4 (4096x384x1024) runs slower than every square case on the same
32x7 tile. `shape_breakdown` puts the shape-side differences into numbers without
board time: the blocking the driver picks, how many times each packed buffer is
reused, how much packing each flop pays for, and how the columns of one C tile land
in the L1 cache when ldc = M (the driver's own choice, `lda = ldc = m`).

Expected values below are worked by hand from the addresses, or read off the
blocking a board run printed (`logs/run_2026-09-24_143441`, "cache blocking
mc:768 nc:384 kc:128" for B4) -- not recomputed the way the module does.
"""

from shape_breakdown import c_tile_lines_per_set, shape_breakdown


def test_b4_c_tile_columns_all_land_in_one_l1_set_pair():
    # ldc = 4096 floats = 16 KiB; the L1 repeats every 128 sets x 64 B = 8 KiB, so
    # each of the 7 columns starts in set 0 and covers sets 0 and 1 (128 B = 2 lines).
    assert c_tile_lines_per_set(ld=4096, mr=32, nr=7) == 7


def test_s1_c_tile_columns_spread_over_separate_sets():
    # ldc = 1000 floats = 4000 B: column j starts at line 62.5*j, so the 7 columns
    # land in sets {0,1} {62,63,64} {125,126} {59,60,61} {122,123} {56,57,58} {119,120}.
    assert c_tile_lines_per_set(ld=1000, mr=32, nr=7) == 1


def test_b1_c_tile_fits_in_l1_ways():
    # ldc = 1024 floats = 4 KiB, half the 8 KiB period: columns 0,2,4 share sets
    # {0,1}, columns 1,3,5 share sets {64,65}. 3 lines per set, under 4 ways.
    assert c_tile_lines_per_set(ld=1024, mr=32, nr=6) == 3


def test_b4_breakdown():
    b = shape_breakdown(m=4096, n=384, k=1024, mr=32, nr=7)
    # Blocking as the board binary printed it for B4.
    assert (b.mc, b.nc, b.kc) == (768, 384, 128)
    # One jc block (nc = N), so A is packed exactly once.
    assert b.a_packs == 1
    # Each packed Bc micro-panel is used by mc/mr = 24 tiles (issue #3's "reuse 24"),
    # each packed Ac micro-panel by ceil(384/7) = 55 tiles.
    assert b.bc_reuse == 24
    assert b.ac_reuse == 55
    # Packed elements per 1000 flops: A once (M*K) + B once (K*N) over 2*M*N*K
    # = 1000 * (1/(2*384) + 1/(2*4096)) = 1.30208 + 0.12207.
    assert abs(b.pack_per_kflop - 1.42415) < 1e-5
    assert b.c_lines_per_set == 7
    assert b.l1_ways == 4
