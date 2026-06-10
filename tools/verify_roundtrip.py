#!/usr/bin/env python3
"""Compare direct.vmp vs roundtrip.vmp per world.

VMP layout: per row of the 2^px-wide map, `width` height bytes then `width` meta bytes.
Meta bits (8-terrain worlds): 0-1 delta half, 2 spare, 3-5 terrain, 6 dual flag, 7 shadow.
The PNG-layer route drops shadow (bit 7) and spare (bit 2) by design; everything else
must round-trip exactly, except dual cells with delta==0 which collapse to single.
"""
import sys
import numpy as np

worlds = sys.argv[1:]
ok = True
for w in worlds:
    a = np.fromfile(f"data/{w}/direct.vmp", dtype=np.uint8)
    b = np.fromfile(f"data/{w}/roundtrip.vmp", dtype=np.uint8)
    if a.size != b.size:
        print(f"{w}: SIZE MISMATCH {a.size} vs {b.size}")
        ok = False
        continue
    n = a.size // 2
    width = 2048
    rows = a.reshape(-1, 2, width)
    ha, ma = rows[:, 0, :].ravel(), rows[:, 1, :].ravel()
    rows = b.reshape(-1, 2, width)
    hb, mb = rows[:, 0, :].ravel(), rows[:, 1, :].ravel()

    hdiff = int((ha != hb).sum())
    bitcounts = [int(((ma ^ mb) >> i & 1).sum()) for i in range(8)]
    # expected lossy: shadow bit 7, spare bit 2, and dual-with-zero-delta collapse
    pair_meta = ma.reshape(-1, 2)
    dual_zero = int(
        (
            (pair_meta[:, 0] & 0x40 != 0)
            & (pair_meta[:, 0] & 0x3 == 0)
            & (pair_meta[:, 1] & 0x3 == 0)
        ).sum()
    )
    essential = hdiff + sum(bitcounts[i] for i in (0, 1, 3, 4, 5, 6))
    status = "OK" if essential == 0 else f"LOSSY (dual-zero-delta pairs: {dual_zero})"
    print(
        f"{w:12s} texels={n:9d} hdiff={hdiff:8d} "
        f"metabits[0..7]={bitcounts} -> {status}"
    )
    if essential != 0 and essential > dual_zero * 4:
        ok = False

print("VERDICT:", "PASS" if ok else "FAIL")
