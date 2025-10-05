import sys
import multiprocessing
import shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    print(f"seg unique before:{np.unique(seg)}, seg shape:{seg.shape}")
    seg[seg>0] = 1
    print(f"seg unique after: {np.unique(seg)}")
    # if seg.max() == 0:
    #     print(f"blank mask will be filterd")
    #     return
    # seg[seg > 0] = 1
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


def prepare_mitolab_2d_training_data(
    source="/mmfs1/data/liupen/project/dataset/mito/mitolab", 
    dst="/mmfs1/data/liupen/project/dataset/mito/training_data_2d"
):
    """
        filter and move mitolab data to a new folder with other 2d data
    """
    valid_ids = subfiles(join(source, 'images'), join=False, suffix='png')
    for v in tqdm(valid_ids):
        input_img = join(source,"images",v)
        input_seg = join(source, "masks", v)
        seg = io.imread(input_seg)
        if seg.max()==0:
            continue 
        output_seg = join(dst, "masks", v)
        output_img = join(dst, "images", v)
        shutil.copy(input_seg, output_seg)
        shutil.copy(input_img, output_img)


def prepare_3dem_2d_training_data(
    source="/mmfs1/data/liupen/project/dataset/mito/2d_slices_of_3dem",
    cell_list=["hela_cell_em","c_elegans_em","fly_brain_em","lucchi_pp_em","glycolytic_muscle_em","IN7gVe4r71le6sTf1UG3__250_300_em"],
    dst="/mmfs1/data/liupen/project/dataset/mito/training_data_2d"
):
    """
    move 3d benchmark data to the 2d training folder
    """
    for cell in cell_list:
        print(f"Processing cell {cell}")
        valid_ids = subfiles(join(source, cell, 'img_x'), join=False, suffix='jpg')
        for v in tqdm(valid_ids):
            input_img = join(source, cell, "img_x", v)
            input_seg = join(source, cell, "label_x", v.replace(".jpg",".png"))
            save_name = cell+"_" + v.replace(".jpg",".png")
            output_seg = join(dst,  "masks", save_name)
            output_img = join(dst, "images", save_name)
            shutil.copy(input_img, output_img)
            shutil.copy(input_seg, output_seg)           

if __name__ == "__main__":
    #prepare_mitolab_2d_training_data()
    #prepare_3dem_2d_training_data()
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/mmfs1/data/liupen/project/dataset/mito/training_data_2d'

    dataset_name = 'Dataset002_MitoSeg'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)


    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(source, 'images'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'images', v),
                         join(source, 'masks', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'EM'}, {'background': 0, 'mitochondria': 1},
                          num_train, '.png', dataset_name=dataset_name)
