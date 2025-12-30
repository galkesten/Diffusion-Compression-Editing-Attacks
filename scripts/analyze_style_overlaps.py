import os
import json
import glob
import pandas as pd

def create_index_coeff_pairs(vector_indices, coefficients):
    """Create set of (index, coeff) tuples from indices and coefficients lists."""
    return set(zip(vector_indices, coefficients))

def analyze_style_overlaps(original_dir, lego_dir, popart_dir):
    """
    Analyze overlaps between original images and their styled versions (LEGO and Pop Art).
    
    For each image, compares:
    - Original vs LEGO
    - Original vs Pop Art
    - Original vs LEGO vs Pop Art (all three)
    
    Calculates intersections and IOU for each comparison at each step,
    then averages across all images.
    
    Args:
        original_dir: Directory containing original step_info JSON files
        lego_dir: Directory containing LEGO style step_info JSON files
        popart_dir: Directory containing Pop Art style step_info JSON files
    
    Returns:
        DataFrame with averaged results per step
    """
    print(f"Searching for step_info JSON files...")
    print(f"  Original: {original_dir}")
    print(f"  LEGO: {lego_dir}")
    print(f"  Pop Art: {popart_dir}")
    
    # Find all original step_info files
    original_files = sorted(glob.glob(os.path.join(original_dir, '*_step_info.json')))
    
    if not original_files:
        raise FileNotFoundError(f"No step_info JSON files found in {original_dir}")
    
    print(f"\nFound {len(original_files)} original images")
    
    # Collect results for each image
    all_results = []
    
    for orig_file in original_files:
        # Extract base name (e.g., "1" from "1_step_info.json")
        base_name = os.path.basename(orig_file).replace('_step_info.json', '')
        
        # Find corresponding LEGO and Pop Art files
        lego_file = os.path.join(lego_dir, f"{base_name}_LEGO_step_info.json")
        popart_file = os.path.join(popart_dir, f"{base_name}_Pop_Art_step_info.json")
        
        # Check if files exist
        if not os.path.exists(lego_file):
            print(f"  Warning: LEGO file not found for {base_name}, skipping...")
            continue
        if not os.path.exists(popart_file):
            print(f"  Warning: Pop Art file not found for {base_name}, skipping...")
            continue
        
        # Load all three JSON files
        with open(orig_file, 'r') as f:
            orig_data = json.load(f)
        with open(lego_file, 'r') as f:
            lego_data = json.load(f)
        with open(popart_file, 'r') as f:
            popart_data = json.load(f)
        
        # Get number of steps (should be same for all)
        num_steps = min(
            orig_data['total_steps'],
            lego_data['total_steps'],
            popart_data['total_steps']
        )
        
        # Analyze each step
        for step_idx in range(num_steps):
            # Get step info for each version
            orig_step = orig_data['steps'][step_idx]
            lego_step = lego_data['steps'][step_idx]
            popart_step = popart_data['steps'][step_idx]
            
            # Create (index, coeff) sets
            orig_set = create_index_coeff_pairs(orig_step['vector_indices'], orig_step['coefficients'])
            lego_set = create_index_coeff_pairs(lego_step['vector_indices'], lego_step['coefficients'])
            popart_set = create_index_coeff_pairs(popart_step['vector_indices'], popart_step['coefficients'])
            
            # Calculate intersections
            orig_lego_intersection = orig_set & lego_set
            orig_popart_intersection = orig_set & popart_set
            all_three_intersection = orig_set & lego_set & popart_set
            
            # Calculate unions
            orig_lego_union = orig_set | lego_set
            orig_popart_union = orig_set | popart_set
            all_three_union = orig_set | lego_set | popart_set
            
            # Calculate IOU
            orig_lego_iou = len(orig_lego_intersection) / len(orig_lego_union) if len(orig_lego_union) > 0 else 0.0
            orig_popart_iou = len(orig_popart_intersection) / len(orig_popart_union) if len(orig_popart_union) > 0 else 0.0
            all_three_iou = len(all_three_intersection) / len(all_three_union) if len(all_three_union) > 0 else 0.0
            
            # Store results
            all_results.append({
                'image': base_name,
                'step': step_idx,
                'orig_lego_intersection': len(orig_lego_intersection),
                'orig_popart_intersection': len(orig_popart_intersection),
                'all_three_intersection': len(all_three_intersection),
                'orig_lego_iou': orig_lego_iou,
                'orig_popart_iou': orig_popart_iou,
                'all_three_iou': all_three_iou,
                'orig_lego_union_size': len(orig_lego_union),
                'orig_popart_union_size': len(orig_popart_union),
                'all_three_union_size': len(all_three_union)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        raise ValueError("No valid comparisons found. Check that all three directories have matching files.")
    
    # Calculate averages per step across all images
    avg_results = df.groupby('step').agg({
        'orig_lego_intersection': 'mean',
        'orig_popart_intersection': 'mean',
        'all_three_intersection': 'mean',
        'orig_lego_iou': 'mean',
        'orig_popart_iou': 'mean',
        'all_three_iou': 'mean',
        'orig_lego_union_size': 'mean',
        'orig_popart_union_size': 'mean',
        'all_three_union_size': 'mean'
    }).reset_index()
    
    # Round to 4 decimal places
    for col in avg_results.columns:
        if col != 'step':
            avg_results[col] = avg_results[col].round(4)
    
    # Save results
    output_csv = os.path.join(original_dir, 'style_overlap_analysis.csv')
    avg_results.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"\nAveraged results across {len(df['image'].unique())} images:")
    print("\n" + avg_results.to_string(index=False))
    
    return avg_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze overlaps between original and styled images')
    parser.add_argument('--original_dir', type=str, default='dataset_Kodack24_compressed',
                       help='Directory containing original step_info JSON files')
    parser.add_argument('--lego_dir', type=str, default='dataset_Kodack24_LEGO_compressed',
                       help='Directory containing LEGO style step_info JSON files')
    parser.add_argument('--popart_dir', type=str, default='dataset_Kodack24_Pop_Art_compressed',
                       help='Directory containing Pop Art style step_info JSON files')
    
    args = parser.parse_args()
    
    analyze_style_overlaps(args.original_dir, args.lego_dir, args.popart_dir)

