#!/usr/bin/env python3
"""
Calculate M value needed to achieve target BPP (bits per pixel).

WHAT IS NBS?
-----------
NBS = "No Bits Steps" - The last NBS steps in the diffusion process where 
NO bits are encoded. These are pure denoising steps without compression.

The BPP formula is:
  bits = (T - NBS - 1) * (bits_for_rank + M * C)
  where bits_for_rank = ceil(log2(comb(K, M)))
  
So NBS reduces the number of steps that need encoding, which reduces total bits and BPP.

WHAT DOES THIS CODE DO?
---------------------
1. For each target BPP (0.1 and 0.05), it tries different M values
2. For each M, it calculates the actual BPP using the formula above
3. It finds the M value that gives BPP closest to the target
4. Uses fixed NBS=1 (as requested)

Uses the turbo_ddcm_bpp function to find M values that achieve target BPP.
"""

import math
import turbo_ddcm.utils as utils

def calculate_bpp_with_nbs(T, K, M, C, img_height, img_width, nbs=1):
    """
    Calculate BPP with fixed NBS value.
    """
    bpp = utils.turbo_ddcm_bpp(T, K, M, C, NBS=nbs, img_height=img_height, img_width=img_width)
    return bpp, nbs

def find_m_for_target_bpp(T, K, C, target_bpp, img_height=512, img_width=512, max_m=None, nbs=1):
    """
    Find M value that achieves target BPP (or closest to it).
    
    Uses binary search to find the best M value.
    """
    if max_m is None:
        max_m = min(K, 1000)  # Reasonable upper limit
    
    best_m = None
    best_bpp_diff = float('inf')
    best_bpp = None
    best_nbs = None
    
    # Try M values from 1 to max_m
    print(f"\nSearching for M to achieve BPP ~ {target_bpp} (with NBS={nbs})...")
    print(f"Trying M values from 1 to {max_m}...")
    
    for m in range(1, max_m + 1):
        try:
            bpp, nbs_val = calculate_bpp_with_nbs(T, K, m, C, img_height, img_width, nbs=nbs)
            diff = abs(bpp - target_bpp)
            
            if diff < best_bpp_diff:
                best_bpp_diff = diff
                best_m = m
                best_bpp = bpp
                best_nbs = nbs_val
            
            # Print progress for every 10th value or when close to target
            if m % 10 == 0 or diff < 0.01:
                print(f"  M={m:4d}: BPP={bpp:.6f}, NBS={nbs_val}, diff={diff:.6f}")
                
        except (OverflowError, ValueError) as e:
            # math.comb(K, M) too large
            print(f"  M={m}: Cannot compute comb({K}, {m}) - stopping search")
            break
    
    return best_m, best_bpp, best_nbs, best_bpp_diff

def main():
    print("="*70)
    print("Calculating M values for target BPP")
    print("="*70)
    
    # Case 1: T=100, K=4096, C=1
    print("\n" + "="*70)
    print("Case 1: T=100, K=4096, C=1, Image size: 512x512")
    print("="*70)
    
    T1, K1, C1 = 100, 4096, 1
    target_bpps = [0.1, 0.05]
    
    for target_bpp in target_bpps:
        print(f"\n{'='*70}")
        print(f"Target BPP: {target_bpp}")
        print(f"{'='*70}")
        m, bpp, nbs, diff = find_m_for_target_bpp(T1, K1, C1, target_bpp, max_m=500, nbs=1)
        if m:
            print(f"\nResult for BPP ~ {target_bpp}:")
            print(f"  M = {m}")
            print(f"  Actual BPP = {bpp:.6f}")
            print(f"  NBS = {nbs}")
            print(f"  Difference from target = {diff:.6f}")
        else:
            print(f"\nCould not find suitable M value")
    
    # Case 2: T=1000, K=256, C=1
    print("\n\n" + "="*70)
    print("Case 2: T=1000, K=256, C=1, Image size: 512x512")
    print("="*70)
    
    T2, K2, C2 = 1000, 256, 1
    
    for target_bpp in target_bpps:
        print(f"\n{'='*70}")
        print(f"Target BPP: {target_bpp}")
        print(f"{'='*70}")
        m, bpp, nbs, diff = find_m_for_target_bpp(T2, K2, C2, target_bpp, max_m=200, nbs=1)
        if m:
            print(f"\nResult for BPP ~ {target_bpp}:")
            print(f"  M = {m}")
            print(f"  Actual BPP = {bpp:.6f}")
            print(f"  NBS = {nbs}")
            print(f"  Difference from target = {diff:.6f}")
        else:
            print(f"\nCould not find suitable M value")
    
    # Case 3: T=1000, K=512, C=1
    print("\n\n" + "="*70)
    print("Case 3: T=1000, K=512, C=1, Image size: 512x512")
    print("="*70)
    
    T3, K3, C3 = 1000, 512, 1
    
    for target_bpp in target_bpps:
        print(f"\n{'='*70}")
        print(f"Target BPP: {target_bpp}")
        print(f"{'='*70}")
        m, bpp, nbs, diff = find_m_for_target_bpp(T3, K3, C3, target_bpp, max_m=400, nbs=1)
        if m:
            print(f"\nResult for BPP ~ {target_bpp}:")
            print(f"  M = {m}")
            print(f"  Actual BPP = {bpp:.6f}")
            print(f"  NBS = {nbs}")
            print(f"  Difference from target = {diff:.6f}")
        else:
            print(f"\nCould not find suitable M value")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

if __name__ == '__main__':
    main()

