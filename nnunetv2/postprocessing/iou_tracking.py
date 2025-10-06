#!/usr/bin/env python3
"""
3D Binary Segmentation to Instance Segmentation via IOU Tracking

This script converts a 3D binary segmentation into a 3D instance segmentation
by tracking connected components across slices using Intersection-over-Union (IoU).

Usage:
    python binary_iou_tracking.py <input_binary.tif> <output_instances.tif> \
        --axis xy --iou_threshold 0.3 --min_size 100

Author: Adapted from empanada-napari codebase
"""

import argparse
import numpy as np
import tifffile
from tqdm import tqdm
from scipy import ndimage
from typing import Dict, List, Tuple

# Import empanada utilities
from empanada.inference.tracker import InstanceTracker
from empanada.inference.matcher import RLEMatcher
from empanada.array_utils import numpy_fill_instances, rle_encode
from empanada.inference import filters


def filter_by_max_size(tracker, max_size):
    """Remove instances larger than max_size voxels"""
    if max_size is None or max_size <= 0:
        return
    
    instance_ids = list(tracker.instances.keys())
    removed_count = 0
    for instance_id in instance_ids:
        instance_attrs = tracker.instances[instance_id]
        size = instance_attrs['runs'].sum()
        
        if size > max_size:
            del tracker.instances[instance_id]
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Removed {removed_count} instances larger than {max_size} voxels")


def filter_touching_border(tracker, volume_shape):
    """Remove instances touching the volume borders"""
    instance_ids = list(tracker.instances.keys())
    removed_count = 0
    
    for instance_id in instance_ids:
        instance_attrs = tracker.instances[instance_id]
        box = instance_attrs['box']
        
        # Check if bounding box touches any border
        touches_border = (
            box[0] == 0 or box[1] == 0 or box[2] == 0 or  # Min borders
            box[3] >= volume_shape[0] or  # Max Z
            box[4] >= volume_shape[1] or  # Max Y
            box[5] >= volume_shape[2]     # Max X
        )
        
        if touches_border:
            del tracker.instances[instance_id]
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Removed {removed_count} instances touching borders")


def filter_by_aspect_ratio(tracker, max_aspect_ratio):
    """Remove instances with aspect ratio > max_aspect_ratio"""
    if max_aspect_ratio is None or max_aspect_ratio <= 0:
        return
    
    instance_ids = list(tracker.instances.keys())
    removed_count = 0
    
    for instance_id in instance_ids:
        instance_attrs = tracker.instances[instance_id]
        box = instance_attrs['box']
        
        # Calculate spans
        zspan = box[3] - box[0]
        yspan = box[4] - box[1]
        xspan = box[5] - box[2]
        
        spans = [zspan, yspan, xspan]
        max_span = max(spans)
        min_span = min(spans)
        
        # Calculate aspect ratio
        aspect_ratio = max_span / max(min_span, 1)  # Avoid division by zero
        
        if aspect_ratio > max_aspect_ratio:
            del tracker.instances[instance_id]
            removed_count += 1
    
    if removed_count > 0:
        print(f"  Removed {removed_count} instances with aspect ratio > {max_aspect_ratio}")


def print_instance_statistics(tracker, label="Instances"):
    """Print statistics about instances in the tracker"""
    if len(tracker.instances) == 0:
        print(f"{label}: 0 instances")
        return
    
    sizes = []
    volumes = []
    aspect_ratios = []
    
    for instance_attrs in tracker.instances.values():
        # Size in voxels
        size = instance_attrs['runs'].sum()
        sizes.append(size)
        
        # Bounding box volume
        box = instance_attrs['box']
        zspan = box[3] - box[0]
        yspan = box[4] - box[1]
        xspan = box[5] - box[2]
        volume = zspan * yspan * xspan
        volumes.append(volume)
        
        # Aspect ratio
        spans = [zspan, yspan, xspan]
        aspect_ratio = max(spans) / max(min(spans), 1)
        aspect_ratios.append(aspect_ratio)
    
    sizes = np.array(sizes)
    volumes = np.array(volumes)
    aspect_ratios = np.array(aspect_ratios)
    
    print(f"\n{label} Statistics:")
    print(f"  Count: {len(tracker.instances)}")
    print(f"  Size (voxels):")
    print(f"    Mean: {sizes.mean():.1f}, Median: {np.median(sizes):.1f}")
    print(f"    Min: {sizes.min()}, Max: {sizes.max()}")
    print(f"  Bounding box volume:")
    print(f"    Mean: {volumes.mean():.1f}, Median: {np.median(volumes):.1f}")
    print(f"  Aspect ratio:")
    print(f"    Mean: {aspect_ratios.mean():.2f}, Median: {np.median(aspect_ratios):.2f}")
    print(f"    Max: {aspect_ratios.max():.2f}")


def binary_slice_to_rle(binary_slice: np.ndarray, class_id: int = 1) -> Dict[int, Dict]:
    """
    Convert a 2D binary slice to RLE format with labeled connected components.
    
    Args:
        binary_slice: 2D boolean or binary array
        class_id: Class ID to assign to instances
    
    Returns:
        Dictionary mapping instance IDs to their RLE attributes (box, starts, runs)
    """
    if binary_slice.sum() == 0:
        return {}
    
    # Label connected components
    labeled_slice, num_labels = ndimage.label(binary_slice)
    
    instances = {}
    for label_id in range(1, num_labels + 1):
        # Get binary mask for this instance
        instance_mask = (labeled_slice == label_id)
        
        # Get bounding box
        coords = np.where(instance_mask)
        if len(coords[0]) == 0:
            continue
            
        y_min, y_max = coords[0].min(), coords[0].max() + 1
        x_min, x_max = coords[1].min(), coords[1].max() + 1
        box = (y_min, x_min, y_max, x_max)
        
        # Convert to RLE encoding
        # Flatten the mask and get indices of True values
        flat_indices = np.where(instance_mask.ravel())[0]
        
        if len(flat_indices) > 0:
            # Sort indices (should already be sorted but just to be safe)
            flat_indices = np.sort(flat_indices)
            
            # Run length encode
            starts, runs = rle_encode(flat_indices)
            
            instances[label_id] = {
                'box': box,
                'starts': starts,
                'runs': runs
            }
    
    return instances


def binary_volume_to_instances_iou_tracking(
    binary_volume: np.ndarray,
    axis: str = 'xy',
    iou_threshold: float = 0.3,
    ioa_threshold: float = 0.5,
    min_size: int = 100,
    min_extent: int = 3,
    max_size: int = None,
    remove_border_instances: bool = False,
    max_aspect_ratio: float = None,
    class_id: int = 1,
    label_divisor: int = 1000,
    verbose: bool = True
) -> np.ndarray:
    """
    Convert a 3D binary segmentation to 3D instance segmentation using IoU tracking.
    
    Args:
        binary_volume: 3D binary array (Z, Y, X) or (D, H, W)
        axis: Tracking axis - 'xy', 'xz', or 'yz'
        iou_threshold: IoU threshold for matching instances across slices
        ioa_threshold: IoA (Intersection over Area) threshold for matching
        min_size: Minimum object size in voxels
        min_extent: Minimum extent in any dimension
        max_size: Maximum object size in voxels (None = no limit)
        remove_border_instances: Remove instances touching volume borders
        max_aspect_ratio: Maximum aspect ratio (max_span/min_span) (None = no limit)
        class_id: Class ID to assign
        label_divisor: Label divisor for panoptic segmentation format
        verbose: Print detailed statistics
    
    Returns:
        Instance segmentation volume with unique IDs for each object
    """
    assert axis in ['xy', 'xz', 'yz'], f"axis must be one of ['xy', 'xz', 'yz'], got {axis}"
    
    # Get axis index
    axis_idx = {'xy': 0, 'xz': 1, 'yz': 2}[axis]
    
    # Reorder axes so that the tracking axis is first
    if axis == 'xy':
        volume = binary_volume  # Already in (Z, Y, X) format
    elif axis == 'xz':
        volume = np.moveaxis(binary_volume, 1, 0)  # (Y, Z, X) -> (Z, Y, X)
    else:  # yz
        volume = np.moveaxis(binary_volume, 2, 0)  # (X, Z, Y) -> (Z, Y, X)
    
    num_slices = volume.shape[0]
    original_shape = binary_volume.shape
    current_shape = volume.shape
    
    print(f"Processing {num_slices} slices along {axis} axis...")
    print(f"Volume shape: {current_shape}")
    print(f"IoU threshold: {iou_threshold}, IoA threshold: {ioa_threshold}")
    
    # Initialize tracker and matcher
    tracker = InstanceTracker(
        class_id=class_id,
        label_divisor=label_divisor,
        shape3d=current_shape,
        axis=axis
    )
    
    matcher = RLEMatcher(
        class_id=class_id,
        label_divisor=label_divisor,
        iou_thr=iou_threshold,
        ioa_thr=ioa_threshold
    )
    
    # Forward pass: match instances across slices
    print("Forward pass: matching instances across slices...")
    rle_stack = []
    
    for slice_idx in tqdm(range(num_slices), desc="Forward matching"):
        slice_2d = volume[slice_idx]
        
        # Convert binary slice to RLE instances
        rle_instances = binary_slice_to_rle(slice_2d, class_id)
        
        # Match with previous slice
        if matcher.target_rle is None:
            matcher.initialize_target(rle_instances)
        else:
            rle_instances = matcher(rle_instances)
        
        rle_stack.append(rle_instances)
    
    # Backward pass: propagate labels backward
    print("Backward pass: propagating labels backward...")
    matcher.target_rle = None
    matcher.assign_new = False
    
    for slice_idx in tqdm(range(num_slices - 1, -1, -1), desc="Backward matching"):
        rle_instances = rle_stack[slice_idx]
        
        # Initialize or match
        if matcher.target_rle is None:
            matcher.initialize_target(rle_instances)
        else:
            rle_instances = matcher(rle_instances)
        
        # Update tracker
        tracker.update(rle_instances, slice_idx)
    
    # Finish tracking
    print("Finalizing tracking...")
    tracker.finish()
    
    # Print statistics before filtering
    if verbose:
        print_instance_statistics(tracker, "Before filtering")
    
    # Apply filters
    print(f"\nApplying filters...")
    initial_count = len(tracker.instances)
    
    # Basic filters
    filters.remove_small_objects(tracker, min_size=min_size)
    filters.remove_pancakes(tracker, min_span=min_extent)
    
    # Additional filters
    if max_size is not None and max_size > 0:
        filter_by_max_size(tracker, max_size)
    
    if remove_border_instances:
        filter_touching_border(tracker, binary_volume.shape)
    
    if max_aspect_ratio is not None and max_aspect_ratio > 0:
        filter_by_aspect_ratio(tracker, max_aspect_ratio)
    
    final_count = len(tracker.instances)
    print(f"Filtering complete: {final_count} instances (removed {initial_count - final_count})")
    
    # Print final statistics
    if verbose and final_count > 0:
        print_instance_statistics(tracker, "After filtering")
    
    # Create output volume
    print("Creating instance segmentation volume...")
    instance_volume = np.zeros(current_shape, dtype=np.uint16)
    numpy_fill_instances(instance_volume, tracker.instances)
    
    # Reorder axes back to original
    if axis == 'xy':
        final_volume = instance_volume
    elif axis == 'xz':
        final_volume = np.moveaxis(instance_volume, 0, 1)
    else:  # yz
        final_volume = np.moveaxis(instance_volume, 0, 2)
    
    return final_volume


def multi_axis_consensus(
    binary_volume: np.ndarray,
    axes: List[str] = ['xy', 'xz', 'yz'],
    iou_threshold: float = 0.3,
    ioa_threshold: float = 0.5,
    pixel_vote_thr: int = 2,
    cluster_iou_thr: float = 0.75,
    min_size: int = 100,
    min_extent: int = 3,
    max_size: int = None,
    remove_border_instances: bool = False,
    max_aspect_ratio: float = None,
    class_id: int = 1,
    label_divisor: int = 1000
) -> Dict[str, np.ndarray]:
    """
    Perform multi-axis tracking and create consensus segmentation.
    
    Args:
        binary_volume: 3D binary array
        axes: List of axes to track on
        iou_threshold: IoU threshold for matching
        ioa_threshold: IoA threshold for matching
        pixel_vote_thr: Minimum votes for a voxel in consensus
        cluster_iou_thr: IoU threshold for clustering in consensus
        min_size: Minimum object size
        min_extent: Minimum extent
        class_id: Class ID
        label_divisor: Label divisor
    
    Returns:
        Dictionary with keys for each axis and 'consensus'
    """
    from empanada.consensus import merge_objects_from_trackers
    
    results = {}
    trackers = []
    
    # Track on each axis
    for axis in axes:
        print(f"\n{'='*60}")
        print(f"Processing axis: {axis}")
        print(f"{'='*60}")
        
        # Get axis index
        axis_idx = {'xy': 0, 'xz': 1, 'yz': 2}[axis]
        
        # Reorder axes
        if axis == 'xy':
            volume = binary_volume
        elif axis == 'xz':
            volume = np.moveaxis(binary_volume, 1, 0)
        else:  # yz
            volume = np.moveaxis(binary_volume, 2, 0)
        
        num_slices = volume.shape[0]
        current_shape = volume.shape
        
        # Initialize tracker and matcher
        tracker = InstanceTracker(
            class_id=class_id,
            label_divisor=label_divisor,
            shape3d=binary_volume.shape,
            axis=axis
        )
        
        matcher = RLEMatcher(
            class_id=class_id,
            label_divisor=label_divisor,
            iou_thr=iou_threshold,
            ioa_thr=ioa_threshold
        )
        
        # Forward pass
        rle_stack = []
        for slice_idx in tqdm(range(num_slices), desc=f"Forward {axis}"):
            slice_2d = volume[slice_idx]
            rle_instances = binary_slice_to_rle(slice_2d, class_id)
            
            if matcher.target_rle is None:
                matcher.initialize_target(rle_instances)
            else:
                rle_instances = matcher(rle_instances)
            
            rle_stack.append(rle_instances)
        
        # Backward pass
        matcher.target_rle = None
        matcher.assign_new = False
        
        for slice_idx in tqdm(range(num_slices - 1, -1, -1), desc=f"Backward {axis}"):
            rle_instances = rle_stack[slice_idx]
            
            # Initialize or match
            if matcher.target_rle is None:
                matcher.initialize_target(rle_instances)
            else:
                rle_instances = matcher(rle_instances)
            
            tracker.update(rle_instances, slice_idx)
        
        tracker.finish()
        trackers.append(tracker)
        
        # Create single-axis result
        instance_volume = np.zeros(binary_volume.shape, dtype=np.uint16)
        numpy_fill_instances(instance_volume, tracker.instances)
        results[axis] = instance_volume
        
        print(f"Axis {axis}: {len(tracker.instances)} instances")
    
    # Create consensus
    if len(trackers) > 1:
        print(f"\n{'='*60}")
        print("Creating consensus from all axes...")
        print(f"{'='*60}")
        
        consensus_instances = merge_objects_from_trackers(
            trackers,
            pixel_vote_thr=pixel_vote_thr,
            cluster_iou_thr=cluster_iou_thr,
            bypass=False
        )
        
        # Apply filters
        consensus_tracker = InstanceTracker(class_id, label_divisor, binary_volume.shape, 'xy')
        consensus_tracker.instances = consensus_instances
        
        initial_count = len(consensus_tracker.instances)
        filters.remove_small_objects(consensus_tracker, min_size=min_size)
        filters.remove_pancakes(consensus_tracker, min_span=min_extent)
        
        if max_size is not None and max_size > 0:
            filter_by_max_size(consensus_tracker, max_size)
        
        if remove_border_instances:
            filter_touching_border(consensus_tracker, binary_volume.shape)
        
        if max_aspect_ratio is not None and max_aspect_ratio > 0:
            filter_by_aspect_ratio(consensus_tracker, max_aspect_ratio)
        
        final_count = len(consensus_tracker.instances)
        if final_count < initial_count:
            print(f"Consensus filtering: {final_count} instances (removed {initial_count - final_count})")
        
        consensus_volume = np.zeros(binary_volume.shape, dtype=np.uint16)
        numpy_fill_instances(consensus_volume, consensus_tracker.instances)
        results['consensus'] = consensus_volume
        
        print(f"Consensus: {len(consensus_tracker.instances)} instances")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D binary segmentation to instance segmentation using IoU tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i","--input_path",
        type=str,
        help="Path to input 3D binary TIFF image"
    )
    
    parser.add_argument(
        "-o","--output_path",
        type=str,
        help="Path to save output instance segmentation TIFF"
    )
    
    parser.add_argument(
        "--axis",
        type=str,
        default="xy",
        choices=['xy', 'xz', 'yz', 'multi'],
        help="Tracking axis (xy=along Z, xz=along Y, yz=along X) or 'multi' for consensus"
    )
    
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for matching instances across slices"
    )
    
    parser.add_argument(
        "--ioa_threshold",
        type=float,
        default=0.5,
        help="IoA threshold for matching instances across slices"
    )
    
    parser.add_argument(
        "--min_size",
        type=int,
        default=500,
        help="Minimum object size in voxels"
    )
    
    parser.add_argument(
        "--min_extent",
        type=int,
        default=3,
        help="Minimum extent (span) in any dimension"
    )
    
    parser.add_argument(
        "--max_size",
        type=int,
        default=None,
        help="Maximum object size in voxels (None = no limit)"
    )
    
    parser.add_argument(
        "--remove_border_instances",
        action="store_true",
        help="Remove instances touching the volume borders"
    )
    
    parser.add_argument(
        "--max_aspect_ratio",
        type=float,
        default=None,
        help="Maximum aspect ratio (max_span/min_span, None = no limit). Useful for filtering elongated artifacts"
    )
    
    parser.add_argument(
        "--pixel_vote_thr",
        type=int,
        default=2,
        help="Pixel vote threshold for multi-axis consensus (only used with --axis multi)"
    )
    
    parser.add_argument(
        "--cluster_iou_thr",
        type=float,
        default=0.75,
        help="Cluster IoU threshold for multi-axis consensus (only used with --axis multi)"
    )
    
    parser.add_argument(
        "--save_all_axes",
        action="store_true",
        help="Save individual axis results when using multi-axis mode"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed statistics output"
    )
    
    args = parser.parse_args()
    
    # Load binary volume
    print(f"Loading binary volume from {args.input_path}...")
    binary_volume = tifffile.imread(args.input_path)
    
    if binary_volume.ndim != 3:
        raise ValueError(f"Input must be 3D, got shape {binary_volume.shape}")
    
    # Convert to binary if needed
    if binary_volume.dtype != bool:
        binary_volume = binary_volume > 0
    
    print(f"Input shape: {binary_volume.shape}")
    print(f"Binary volume statistics: {binary_volume.sum()} positive voxels "
          f"({100 * binary_volume.sum() / binary_volume.size:.2f}%)")
    
    # Process based on axis mode
    if args.axis == 'multi':
        print("\nUsing multi-axis consensus mode...")
        results = multi_axis_consensus(
            binary_volume,
            axes=['xy', 'xz', 'yz'],
            iou_threshold=args.iou_threshold,
            ioa_threshold=args.ioa_threshold,
            pixel_vote_thr=args.pixel_vote_thr,
            cluster_iou_thr=args.cluster_iou_thr,
            min_size=args.min_size,
            min_extent=args.min_extent,
            max_size=args.max_size,
            remove_border_instances=args.remove_border_instances,
            max_aspect_ratio=args.max_aspect_ratio
        )
        
        # Save consensus result
        output_volume = results['consensus']
        print(f"\nSaving consensus result to {args.output_path}...")
        tifffile.imwrite(args.output_path, output_volume.astype(np.uint16), compression='zlib')
        
        # Optionally save individual axis results
        if args.save_all_axes:
            import os
            base_name = args.output_path.rsplit('.', 1)[0]
            for axis_name in ['xy', 'xz', 'yz']:
                axis_output = f"{base_name}_{axis_name}.tif"
                print(f"Saving {axis_name} result to {axis_output}...")
                tifffile.imwrite(axis_output, results[axis_name].astype(np.uint16), compression='zlib')
    else:
        # Single axis tracking
        output_volume = binary_volume_to_instances_iou_tracking(
            binary_volume,
            axis=args.axis,
            iou_threshold=args.iou_threshold,
            ioa_threshold=args.ioa_threshold,
            min_size=args.min_size,
            min_extent=args.min_extent,
            max_size=args.max_size,
            remove_border_instances=args.remove_border_instances,
            max_aspect_ratio=args.max_aspect_ratio,
            verbose=not args.quiet
        )
        
        print(f"\nSaving result to {args.output_path}...")
        tifffile.imwrite(args.output_path, output_volume.astype(np.uint16), compression='zlib')
    
    # Print statistics
    num_instances = len(np.unique(output_volume)) - 1  # Exclude background
    print(f"\nFinal statistics:")
    print(f"  Number of instances: {num_instances}")
    print(f"  Output shape: {output_volume.shape}")
    print(f"  Output dtype: {output_volume.dtype}")
    print(f"  Max instance ID: {output_volume.max()}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

