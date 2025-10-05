from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import tifffile
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from pathlib import Path


def convert_mito3d(mito_base_dir: str, nnunet_dataset_id: int = 220):
    task_name = "MitoSeg3D"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    out_dir = Path(nnUNet_raw.replace('"', "")) / foldername

    cases = subfiles(join(mito_base_dir, 'images'), join=False, suffix='tiff')
    num_training_cases = len(cases)
    print(f"Cases: {cases}")
    for tr in cases:
        shutil.copy(join(mito_base_dir, "images", tr), join(imagestr, f'{tr}_0000.tiff'))
        # convert mask to binary mask
        mask = tifffile.imread(join(mito_base_dir, "masks", tr.replace('image', 'mask')))
        mask[mask > 0] = 1
        tifffile.imwrite(join(labelstr, f'{tr}.tiff'), mask)
        #shutil.copy(join(mito_base_dir, "masks", tr.replace('image', 'mask')), join(labelstr, f'{tr}.tiff'))

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "EM",
        },
        labels={
            "background": 0,
            "mitochondria": 1,
        },
        file_ending=".tiff",
        num_training_cases=num_training_cases,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="/mmfs1/data/liupen/project/MitoSeg/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoSeg3D",
                        help="The downloaded and extracted KiTS2023 dataset (must have case_XXXXX subfolders)")
    parser.add_argument('-d', required=False, type=int, default=3, help='nnU-Net Dataset ID, default: 220')
    args = parser.parse_args()
    amos_base = args.input_folder
    convert_mito3d(amos_base, args.d)

    # /media/isensee/raw_data/raw_datasets/kits23/dataset

