import argparse
import os
import numpy as np
from typing import Union, Tuple, Optional
import SimpleITK as sitk
import tifffile as tiff
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_dilation, ball
from skimage.measure import label
from skimage.filters import gaussian
import cc3d


def load_image(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load 3D image from file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (image_array, affine_matrix). affine_matrix is None for TIFF files.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.gz':
        # Handle .nii.gz files
        file_ext = os.path.splitext(os.path.splitext(file_path)[0])[1].lower()
        if file_ext == '.nii':
            # Use SimpleITK for .nii.gz files
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)
            affine_matrix = np.array(image.GetDirection() + image.GetOrigin()).reshape(4, 4)
            return image_array, affine_matrix
    elif file_ext == '.tiff' or file_ext == '.tif':
        # Use tifffile for TIFF files
        image_array = tiff.imread(file_path)
        return image_array, None
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .nii.gz, .tiff, .tif")


def save_image(image_array: np.ndarray, file_path: str, affine_matrix: Optional[np.ndarray] = None):
    """
    Save 3D image to file.
    
    Args:
        image_array: 3D numpy array to save
        file_path: Output file path
        affine_matrix: Affine transformation matrix (for NIfTI files)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.gz':
        # Handle .nii.gz files
        file_ext = os.path.splitext(os.path.splitext(file_path)[0])[1].lower()
        if file_ext == '.nii':
            # Use SimpleITK for .nii.gz files
            image = sitk.GetImageFromArray(image_array)
            if affine_matrix is not None:
                # Set direction and origin from affine matrix
                direction = affine_matrix[:3, :3].flatten()
                origin = affine_matrix[:3, 3]
                image.SetDirection(direction)
                image.SetOrigin(origin)
            sitk.WriteImage(image, file_path)
    elif file_ext == '.tiff' or file_ext == '.tif':
        # Use tifffile for TIFF files
        tiff.imwrite(file_path, image_array, compression='zlib')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .nii.gz, .tiff, .tif")


def watershed_3d(binary_mask: np.ndarray, 
                 min_distance: int = 5,
                 sigma: float = 1.0,
                 min_size: int = 100,
                 connectivity: int = 6) -> np.ndarray:
    """
    Apply watershed segmentation to 3D binary mask.
    
    Args:
        binary_mask: 3D binary mask (0 and 1 values)
        min_distance: Minimum distance between seeds
        sigma: Gaussian blur sigma for distance transform
        min_size: Minimum size of objects to keep
        connectivity: Connectivity for connected components (6 or 26)
        
    Returns:
        3D labeled segmentation mask
    """
    # Ensure binary mask
    binary_mask = (binary_mask == 1).astype(np.uint8)
    
    # Apply Gaussian blur to smooth the mask
    if sigma > 0:
        binary_mask_smooth = gaussian(binary_mask.astype(float), sigma=sigma)
    else:
        binary_mask_smooth = binary_mask.astype(float)
    
    # Create distance transform
    from scipy.ndimage import distance_transform_edt
    distance = distance_transform_edt(binary_mask_smooth)
    
    # Find local maxima as seeds
    from scipy.ndimage import maximum_filter
    local_maxima = maximum_filter(distance, size=min_distance) == distance
    local_maxima = local_maxima & (distance > 0)
    
    # Label the seeds
    seeds = label(local_maxima)
    
    # Apply watershed
    segmentation = watershed(-distance, seeds, mask=binary_mask)
    
    # Remove small objects
    if min_size > 0:
        segmentation = remove_small_objects(segmentation, min_size=min_size)
    
    # Ensure proper connectivity using cc3d
    segmentation = cc3d.connected_components(segmentation, connectivity=connectivity)
    
    return segmentation


def watershed_from_file(input_path: str, 
                       output_path: str,
                       min_distance: int = 5,
                       sigma: float = 1.0,
                       min_size: int = 100,
                       connectivity: int = 6) -> None:
    """
    Apply watershed segmentation to a 3D binary mask file.
    
    Args:
        input_path: Path to input binary mask file
        output_path: Path to output segmentation file
        min_distance: Minimum distance between seeds
        sigma: Gaussian blur sigma for distance transform
        min_size: Minimum size of objects to keep
        connectivity: Connectivity for connected components (6 or 26)
    """
    print(f"Loading image from: {input_path}")
    image_array, affine_matrix = load_image(input_path)
    
    print(f"Image shape: {image_array.shape}")
    print(f"Image dtype: {image_array.dtype}")
    print(f"Image value range: [{image_array.min()}, {image_array.max()}]")
    
    # Apply watershed
    print("Applying watershed segmentation...")
    segmentation = watershed_3d(image_array, min_distance, sigma, min_size, connectivity)
    
    print(f"Segmentation complete. Found {segmentation.max()} objects.")
    
    # Save result
    print(f"Saving result to: {output_path}")
    save_image(segmentation.astype(np.uint16), output_path, affine_matrix)
    print("Done!")


def process_folder(input_folder: str,
                  output_folder: str,
                  file_pattern: str = "*.nii.gz",
                  min_distance: int = 5,
                  sigma: float = 1.0,
                  min_size: int = 100,
                  connectivity: int = 6) -> None:
    """
    Process all files in a folder with watershed segmentation.
    
    Args:
        input_folder: Input folder containing binary mask files
        output_folder: Output folder for segmentation results
        file_pattern: File pattern to match (e.g., "*.nii.gz", "*.tiff"), or "auto" to search all supported formats
        min_distance: Minimum distance between seeds
        sigma: Gaussian blur sigma for distance transform
        min_size: Minimum size of objects to keep
        connectivity: Connectivity for connected components (6 or 26)
    """
    import glob
    
    os.makedirs(output_folder, exist_ok=True)
    
    # If pattern is "auto", search for all supported formats
    if file_pattern == "auto":
        patterns = ["*.nii.gz", "*.tiff", "*.tif"]
        input_files = []
        for pattern in patterns:
            search_pattern = os.path.join(input_folder, pattern)
            input_files.extend(glob.glob(search_pattern))
    else:
        # Find all matching files
        search_pattern = os.path.join(input_folder, file_pattern)
        input_files = glob.glob(search_pattern)
    
    if not input_files:
        print(f"No files found in folder: {input_folder}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)
        if ext == '.gz':
            name = os.path.splitext(name)[0]  # Remove .nii from .nii.gz
            ext = '.nii.gz'  # Keep full extension for output
        
        output_file = os.path.join(output_folder, f"{name}_watershed{ext}")
        
        print(f"\nProcessing: {filename}")
        try:
            watershed_from_file(input_file, output_file, min_distance, sigma, min_size, connectivity)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue


def entry_point_watershed():
    """Command line interface for watershed segmentation."""
    parser = argparse.ArgumentParser(description='Apply watershed segmentation to 3D binary masks')
    
    # Input/Output arguments
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input file or folder path (auto-detected)')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output file or folder path')
    
    # Processing parameters
    parser.add_argument('--min_distance', type=int, default=30,
                       help='Minimum distance between seeds (default: 25)')
    parser.add_argument('--sigma', type=float, default=2.0,
                       help='Gaussian blur sigma for distance transform (default: 1.0)')
    parser.add_argument('--min_size', type=int, default=200,
                       help='Minimum size of objects to keep (default: 200)')
    parser.add_argument('--connectivity', type=int, default=6, choices=[6, 26],
                       help='Connectivity for connected components (default: 6)')
    
    # Folder processing options
    parser.add_argument('--pattern', type=str, default='auto',
                       help='File pattern for folder processing (default: auto - searches for .nii.gz, .tiff, .tif)')
    
    args = parser.parse_args()
    
    # Auto-detect if input is a file or folder
    if os.path.isdir(args.input):
        print(f"Detected folder input: {args.input}")
        print(f"Output folder: {args.output}")
        print(f"File pattern: {args.pattern}")
        process_folder(args.input, args.output, args.pattern, 
                      args.min_distance, args.sigma, args.min_size, args.connectivity)
    elif os.path.isfile(args.input):
        print(f"Detected file input: {args.input}")
        print(f"Output file: {args.output}")
        watershed_from_file(args.input, args.output, 
                           args.min_distance, args.sigma, args.min_size, args.connectivity)
    else:
        raise ValueError(f"Input path does not exist: {args.input}")


if __name__ == '__main__':
    entry_point_watershed()
