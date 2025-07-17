#!/usr/bin/env python3

import os
import argparse

def rename_npy_with_source_names(source_dir, target_dir, dry_run=False):
    # Supported image extensions for source files
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic')
    
    # Get sorted list of source image files
    source_files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and f.lower().endswith(IMAGE_EXTS)
    ], key=lambda x: x.lower())
    
    # Get sorted list of target .npy files
    target_files = sorted([
        f for f in os.listdir(target_dir)
        if os.path.isfile(os.path.join(target_dir, f))
        and f.lower().endswith('.npy')
    ], key=lambda x: x.lower())
    
    # Verify file counts match
    if len(source_files) != len(target_files):
        print(f"Error: File count mismatch. Source has {len(source_files)} images, target has {len(target_files)} .npy files.")
        return False
    
    print(f"Preparing to rename {len(target_files)} .npy files...")
    
    renamed_count = 0
    for src_file, tgt_file in zip(source_files, target_files):
        # Get source name without extension
        src_name = os.path.splitext(src_file)[0]
        
        # Keep .npy extension from target
        new_name = f"{src_name}.npy"
        old_path = os.path.join(target_dir, tgt_file)
        new_path = os.path.join(target_dir, new_name)
        
        if dry_run:
            print(f"[DRY RUN] Would rename: '{tgt_file}' -> '{new_name}'")
        else:
            try:
                # Handle filename conflicts
                if os.path.exists(new_path):
                    counter = 1
                    while os.path.exists(os.path.join(target_dir, f"{src_name}_{counter}.npy")):
                        counter += 1
                    new_name = f"{src_name}_{counter}.npy"
                    new_path = os.path.join(target_dir, new_name)
                
                os.rename(old_path, new_path)
                print(f"Renamed: '{tgt_file}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f"Error: Failed to rename '{tgt_file}' -> '{new_name}': {str(e)}")
    
    if dry_run:
        print("[DRY RUN] No actual changes made")
    else:
        print(f"Operation complete! Successfully renamed {renamed_count}/{len(target_files)} .npy files")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename .npy files in target folder using names from source image files"
    )
    parser.add_argument("source_dir", help="Path to source directory containing image files")
    parser.add_argument("target_dir", help="Path to target directory containing .npy files to be renamed")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without actually renaming")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory does not exist - {args.source_dir}")
        exit(1)
    if not os.path.isdir(args.target_dir):
        print(f"Error: Target directory does not exist - {args.target_dir}")
        exit(1)
    
    rename_npy_with_source_names(args.source_dir, args.target_dir, args.dry_run)