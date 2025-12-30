#!/usr/bin/env python3
"""
Calculate BPP for specific parameters.
"""

import turbo_ddcm.utils as utils

def main():
    print("="*70)
    print("BPP Calculation for Specific Parameters")
    print("="*70)
    
    T = 1000
    C = 1
    M = 1
    img_height = 512
    img_width = 512
    nbs = 1
    
    cases = [
        ("Case 1", 256),
        ("Case 2", 512)
    ]
    
    for case_name, K in cases:
        print(f"\n{case_name}: T={T}, K={K}, M={M}, C={C}, NBS={nbs}")
        print("-" * 70)
        
        bpp = utils.turbo_ddcm_bpp(T, K, M, C, NBS=nbs, img_height=img_height, img_width=img_width)
        
        # Calculate components for explanation
        import math
        bits_for_rank = math.ceil(math.log2(math.comb(K, M)))
        bits_for_coeffs = M * C
        bits_per_step = bits_for_rank + bits_for_coeffs
        num_steps_with_bits = T - nbs - 1
        total_bits = num_steps_with_bits * bits_per_step
        
        print(f"  bits_for_rank (ceil(log2(comb({K}, {M})))) = {bits_for_rank}")
        print(f"  bits_for_coeffs (M * C) = {M} * {C} = {bits_for_coeffs}")
        print(f"  bits_per_step = {bits_for_rank} + {bits_for_coeffs} = {bits_per_step}")
        print(f"  num_steps_with_bits = T - NBS - 1 = {T} - {nbs} - 1 = {num_steps_with_bits}")
        print(f"  total_bits = {num_steps_with_bits} * {bits_per_step} = {total_bits}")
        print(f"  image_pixels = {img_height} * {img_width} = {img_height * img_width}")
        print(f"  BPP = {total_bits} / {img_height * img_width} = {bpp:.8f}")
        print(f"\n  Result: BPP = {bpp:.6f}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

