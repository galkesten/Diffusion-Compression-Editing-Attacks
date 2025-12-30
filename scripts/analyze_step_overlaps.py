import os
import json
import glob
from collections import defaultdict
import pandas as pd

def create_index_coeff_pairs(vector_indices, coefficients):
    return set(zip(vector_indices, coefficients))

def analyze_step_overlaps(compressed_dir):
    print(f"Searching for step_info JSON files in: {compressed_dir}")
    json_files = sorted(glob.glob(os.path.join(compressed_dir, '*_step_info.json')))
    
    if not json_files:
        raise FileNotFoundError(f"No step_info JSON files found in {compressed_dir}")

    print(f"\nFound {len(json_files)} step_info files:")
    for i, json_file in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(json_file)}")

    print(f"\nLoading step info from all files...")
    all_step_data = []
    for json_file in json_files:
        print(f"  Loading: {os.path.basename(json_file)}", end=" ... ")
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_step_data.append(data)
            print(f"OK ({data['total_steps']} steps, M={data.get('M', 'N/A')}, seed={data.get('seed', 'N/A')})")
    

    max_steps = max(data['total_steps'] for data in all_step_data)
    print(f"\nAnalyzing {max_steps} steps across {len(all_step_data)} images")

    # Get all unique timesteps from the first image (assuming all use same schedule)
    # We'll use these as reference timesteps
    reference_timesteps = [step['timestep'] for step in all_step_data[0]['steps']]
    print(f"Reference timesteps: {reference_timesteps[:5]}... (showing first 5)")

    results = []
    
    for step_idx in range(max_steps):
        reference_timestep = reference_timesteps[step_idx] if step_idx < len(reference_timesteps) else None
      
        step_sets = [] 
        step_indices_sets = [] 
        timestep_mismatches = []

        for img_idx, img_data in enumerate(all_step_data):
            if step_idx < len(img_data['steps']):
                step_info = img_data['steps'][step_idx]
                step_timestep = step_info.get('timestep', None)
                
                # Verify timestep matches (warn if not)
                if reference_timestep is not None and step_timestep != reference_timestep:
                    timestep_mismatches.append((img_idx, step_timestep, reference_timestep))
                
                indices = step_info['vector_indices']  
                coeffs = step_info['coefficients']  
                index_coeff_set = create_index_coeff_pairs(indices, coeffs)
                step_sets.append(index_coeff_set)
                
                step_indices_sets.append(set(indices))
            else:
                step_sets.append(set())
                step_indices_sets.append(set())
        
        # Warn if timesteps don't match
        if timestep_mismatches:
            print(f"  Warning: Step {step_idx} has timestep mismatches: {timestep_mismatches[:3]}...")
        
        if step_sets:
            intersection = step_sets[0]
            for s in step_sets[1:]:
                intersection = intersection & s  
            
            union = set()
            for s in step_sets:
                union = union | s
            

            overlap_count = len(intersection)  
            union_count = len(union)  
            iou = overlap_count / union_count if union_count > 0 else 0.0  
            
            shared_indices = sorted([idx for idx, coeff in intersection])
            
            shared_indices_any_coeff = set(step_indices_sets[0])
            for s in step_indices_sets[1:]:
                shared_indices_any_coeff = shared_indices_any_coeff & s
            
            # Calculate maximum number of images that share at least one pair
            # Count how many images each (index, coeff) pair appears in
            pair_counts = {}
            for pair in union:
                count = sum(1 for s in step_sets if pair in s)
                pair_counts[pair] = count
            
            # Find maximum count (maximum number of images sharing a pair)
            # e.g., if max_overlap_count = 3, it means at least one pair appears in all 3 images
            max_overlap_count = max(pair_counts.values()) if pair_counts else 0
            
            # Count how many (index, coeff) pairs achieve that maximum overlap
            # e.g., if max_overlap_count = 3 and pairs_with_max_overlap = 5,
            # it means 5 different (index, coeff) pairs each appear in all 3 images
            pairs_with_max_overlap = sum(1 for count in pair_counts.values() if count == max_overlap_count)
            
            results.append({
                'step': step_idx,
                'timestep': reference_timestep if reference_timestep is not None else 'N/A',
                'overlap_raw_number': overlap_count,
                'iou': round(iou, 6),
                'shared_indices': ','.join(map(str, shared_indices)) if shared_indices else '',
                'shared_indices_count': len(shared_indices),
                'shared_indices_any_coeff': ','.join(map(str, sorted(shared_indices_any_coeff))) if shared_indices_any_coeff else '',
                'shared_indices_any_coeff_count': len(shared_indices_any_coeff),
                'union_size': union_count,
                'intersection_size': overlap_count,
                'max_images_sharing_pair': max_overlap_count,
                'num_pairs_with_max_overlap': pairs_with_max_overlap
            })
            
            timestep_str = f" (t={reference_timestep})" if reference_timestep is not None else ""
            print(f"Step {step_idx}{timestep_str}: Overlap={overlap_count}, IOU={iou:.4f}, Shared indices={len(shared_indices)}, Max images sharing a pair={max_overlap_count}")
    
    df = pd.DataFrame(results)
    output_csv = os.path.join(compressed_dir, 'step_overlap_analysis.csv')
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"\nSummary:")
    print(df[['step', 'overlap_raw_number', 'iou', 'shared_indices_count', 'max_images_sharing_pair', 'num_pairs_with_max_overlap']].to_string())
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze step overlaps across compressed images')
    parser.add_argument('--compressed_dir', type=str, default='dataset_Kodack24_compressed',
                       help='Directory containing step_info JSON files')
    
    args = parser.parse_args()
    
    analyze_step_overlaps(args.compressed_dir)

