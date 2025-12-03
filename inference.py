import os
import time
import argparse


import numpy as np
import rasterio
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter                 # will need scatter GPU support
# from mmengine.dataset import default_collate as collate

# from mmengine.config import Config
from mmseg.apis import init_segmentor
# from mmseg.apis import init_model as init_segmentor
# from mmseg.datasets.pipelines import Compose, LoadImageFromFile
from mmcv.transforms import Compose, LoadImageFromFile
from skimage import exposure

from geospatial_fm.temporal_encoder_decoder import TemporalEncoderDecoder
from geospatial_fm.geospatial_pipelines import LoadGeospatialImageFromFile

# torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

# config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification", 
#                             filename="multi_temporal_crop_classification_Prithvi_100M.py", 
#                             token=os.environ.get("token"))
config_path = "multi_temporal_crop_classification_Prithvi_100M.py"
# ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification", 
#                      filename='multi_temporal_crop_classification_Prithvi_100M.pth', 
#                      token=os.environ.get("token"))
ckpt = "prithvi_local_repo/multi_temporal_crop_classification_Prithvi_100M.pth"
##########

CDL_COLOR_MAP = [{'value': 1, 'label': 'Natural vegetation', 'rgb': (233,255,190)},
                 {'value': 2, 'label': 'Forest', 'rgb': (149,206,147)},
                 {'value': 3, 'label': 'Corn', 'rgb': (255,212,0)},
                 {'value': 4, 'label': 'Soybeans', 'rgb': (38,115,0)},
                 {'value': 5, 'label': 'Wetlands', 'rgb': (128,179,179)},
                 {'value': 6, 'label': 'Developed/Barren', 'rgb': (156,156,156)},
                 {'value': 7, 'label': 'Open Water', 'rgb': (77,112,163)},
                 {'value': 8, 'label': 'Winter Wheat', 'rgb': (168,112,0)},
                 {'value': 9, 'label': 'Alfalfa', 'rgb': (255,168,227)},
                 {'value': 10, 'label': 'Fallow/Idle cropland', 'rgb': (191,191,122)},
                 {'value': 11, 'label': 'Cotton', 'rgb':(255,38,38)},
                 {'value': 12, 'label': 'Sorghum', 'rgb':(255,158,15)},
                 {'value': 13, 'label': 'Other', 'rgb':(0,175,77)}]


def apply_color_map(rgb, color_map=CDL_COLOR_MAP):

    rgb_mapped = rgb.copy()
    
    for map_tmp in color_map:
        
        for i in range(3):
            rgb_mapped[i] = np.where((rgb[0] == map_tmp['value']) & (rgb[1] == map_tmp['value']) & (rgb[2] == map_tmp['value']), map_tmp['rgb'][i], rgb_mapped[i])
    
    return rgb_mapped    
    

def stretch_rgb(rgb):
    
    ls_pct=0
    pLow, pHigh = np.percentile(rgb[~np.isnan(rgb)], (ls_pct,100-ls_pct))
    img_rescale = exposure.rescale_intensity(rgb, in_range=(pLow,pHigh))
    
    return img_rescale

def open_tiff(fname):
    
    with rasterio.open(fname, "r") as src:
        
        data = src.read()
        
    return data

def write_tiff(img_wrt, filename, metadata):

    """
    It writes a raster image to file.

    :param img_wrt: numpy array containing the data (can be 2D for single band or 3D for multiple bands)
    :param filename: file path to the output file
    :param metadata: metadata to use to write the raster to disk
    :return:
    """

    with rasterio.open(filename, "w", **metadata) as dest:

        if len(img_wrt.shape) == 2:
            
            img_wrt = img_wrt[None]

        for i in range(img_wrt.shape[0]):
            dest.write(img_wrt[i, :, :], i + 1)
    
    return filename
            

def get_meta(fname):
    
    with rasterio.open(fname, "r") as src:
        
        meta = src.meta
        
    return meta

def preprocess_example(example_list):
    
    example_list = [os.path.join(os.path.abspath(''), x) for x in example_list]
    
    return example_list


def inference_segmentor(model, imgs, custom_test_pipeline=None):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImageFromFile()] + cfg.data.test.pipeline[1:] if custom_test_pipeline is None else custom_test_pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = {'img_info': {'filename': img}}
        img_data = test_pipeline(img_data)
        data.append(img_data)
    # print(data.shape)
    
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # data = collate(data, samples_per_gpu=len(imgs))
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # img_metas = scatter(data['img_metas'],'cpu')
        # data['img_metas'] = [i.data[0] for i in data['img_metas']]
        
        img_metas = data['img_metas'].data[0]
        img = data['img']
        data = {'img': img, 'img_metas':img_metas}
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def process_rgb(input, mask, indexes):

    rgb = stretch_rgb((input[indexes, :, :].transpose((1,2,0))/10000*255).astype(np.uint8))
    rgb = np.where(mask.transpose((1,2,0)) == 1, 0, rgb)
    rgb = np.where(rgb < 0, 0, rgb)
    rgb = np.where(rgb > 255, 255, rgb)

    return rgb

def inference_on_file(target_image, model, custom_test_pipeline):

    # target_image is already a string path
    time_taken=-1
    st = time.time()
    print('Running inference...')
    result = inference_segmentor(model, target_image, custom_test_pipeline)
    print("Output has shape: " + str(result[0].shape))

    ##### get metadata mask
    input = open_tiff(target_image)
    meta = get_meta(target_image)
    mask = np.where(input == meta['nodata'], 1, 0)
    mask = np.max(mask, axis=0)[None]
    
    rgb1 = process_rgb(input, mask, [2, 1, 0])
    rgb2 = process_rgb(input, mask, [8, 7, 6])
    rgb3 = process_rgb(input, mask, [14, 13, 12])

    result[0] = np.where(mask == 1, 0, result[0])

    et = time.time()
    time_taken = np.round(et - st, 1)
    print(f'Inference completed in {str(time_taken)} seconds')
    
    output=result[0][0] + 1
    output = np.vstack([output[None], output[None], output[None]]).astype(np.uint8)
    output=apply_color_map(output).transpose((1,2,0))
        
    return rgb1,rgb2,rgb3,output

def process_test_pipeline(custom_test_pipeline, bands=None):
    
    # change extracted bands if necessary
    if bands is not None:
        
        extract_index = [i for i, x in enumerate(custom_test_pipeline) if x['type'] == 'BandsExtract' ]
        
        if len(extract_index) > 0:
            
            custom_test_pipeline[extract_index[0]]['bands'] = eval(bands)
            
    collect_index = [i for i, x in enumerate(custom_test_pipeline) if x['type'].find('Collect') > -1]
    
    # adapt collected keys if necessary
    if len(collect_index) > 0:
        
        keys = ['img_info', 'filename', 'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']
        custom_test_pipeline[collect_index[0]]['meta_keys'] = keys
    
    return custom_test_pipeline

config = Config.fromfile(config_path)
config.model.backbone.pretrained=None
model = init_segmentor(config, ckpt, device='cpu')
custom_test_pipeline=process_test_pipeline(model.cfg.data.test.pipeline, None)


# CLI tool for inference

def main():
    parser = argparse.ArgumentParser(description="Run crop type inference on a geotiff image.")
    parser.add_argument("input_image", help="Path to input geotiff image")
    parser.add_argument("output_raster", help="Path to output georeferenced raster")
    args = parser.parse_args()

    # Run inference
    rgb1, rgb2, rgb3, output = inference_on_file(args.input_image, model, custom_test_pipeline)

    # Get metadata from input image
    meta = get_meta(args.input_image)
    # Write output raster
    write_tiff(output, args.output_raster, meta)
    print(f"Output written to {args.output_raster}")

if __name__ == "__main__":
    main()