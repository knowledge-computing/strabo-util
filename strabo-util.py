import os
from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from skimage import io, util, morphology, measure
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import resize


@contextmanager
def all_process(imagery_fn, outputfile='', resize_ratio=0.1, log_on=False, directory=''):
    # resize for fast processing
    # resize_ratio = 0.1
    imagery_fn = str(directory) + imagery_fn
    # for full size images
    small_objects_size = 6000
    dilation_combine_edges = 15
    dilation_remove_frame = 60

    small_objects_size = int(small_objects_size * resize_ratio)
    dilation_combine_edges = int(dilation_combine_edges * resize_ratio)
    dilation_remove_frame = int(dilation_remove_frame * resize_ratio)

    # downsample GeoTIFF
    with rasterio.open(imagery_fn) as src:
        t = src.transform

        # rescale the metadata
        transform = Affine(t.a / resize_ratio, t.b, t.c, t.d, t.e / resize_ratio, t.f)
        height = int(src.height * resize_ratio)
        width = int(src.width * resize_ratio)

        profile = src.profile
        profile.update(transform=transform, driver='GTiff', height=height, width=width)

        data = src.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
            out_shape=(src.count, height, width),
            resampling=Resampling.bilinear,
        )

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:  # Open as DatasetWriter
                dataset.write(data)
                del data

            with memfile.open() as dataset:  # Reopen as DatasetReader
                data = dataset.read(
                    out_shape=(dataset.count, int(dataset.height), int(dataset.width))
                )
            height = src.height
            width = src.width
            profile = src.profile
            image = reshape_as_image(data)
            image = rgb2gray(image)
            src_data = src.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
                out_shape=(src.count, src.height, src.width)
            )
            if log_on:
                io.imsave('1input.png', image)
            # extract map collar
            img_collar = np.where(image == 1, 1, 0)

            # detect Canny edges
            sedges = canny(image)
            if log_on:
                io.imsave('2edges.png', sedges)
            # clean edges
            # combine nearby edges
            mask = morphology.disk(dilation_combine_edges)
            sedges = morphology.dilation(sedges, selem=mask)
            if log_on:
                io.imsave('3edges_combined.png', sedges)

            # remove edge noise
            sedges_remove = morphology.remove_small_objects(sedges, small_objects_size, connectivity=2)
            if log_on:
                io.imsave('4edges_cleaned.png', sedges_remove)

            # detect the largest no-edge area
            sedges = util.invert(sedges_remove)
            blobs_labels = measure.label(sedges)
            assert (blobs_labels.max() != 0)  # assume at least 1 CC
            target_area = blobs_labels == np.argmax(np.bincount(blobs_labels.flat)[1:]) + 1
            if log_on:
                io.imsave('5largestCC.png', target_area)

            # clean target_area
            target_area = morphology.remove_small_holes(target_area, connectivity=2)
            if log_on:
                io.imsave('6largestCCRemoveHoles.png', target_area)

            # detect frames
            mask = morphology.disk(dilation_remove_frame)
            frame = np.logical_and(morphology.dilation(target_area, selem=mask),
                                   morphology.dilation(img_collar, selem=mask))
            if log_on:
                io.imsave('7frame.png', frame)

            # combine everything
            target_area = np.logical_or(np.logical_or(target_area, frame), img_collar)
            if log_on:
                io.imsave('8largestCCFinal.png', target_area)

            # create GeoTIFF mask
            mask = resize(target_area, (height, width), anti_aliasing=False)
            src_data = np.where(mask == 1, 255, src_data)

            # src_data = np.where(target_area == 1, 255, data)

            if not outputfile:
                outputfile = np.os.path.splitext(imagery_fn)[0] + '_mask.tif'
            else:
                outputfile = directory + outputfile
            # save GeoTIFF mask
            with rasterio.open(
                    outputfile,
                    'w',
                    **profile
            ) as dst:
                dst.write(src_data)


# def main(argv):
#     inputfile = ''
#     outputfile = ''
#     log_on = False
#     resize_ratio = 0.1
#     try:
#         opts, args = getopt.getopt(argv, 'i:o:r:')
#     except getopt.GetoptError:
#         print('strabo-util.py -i <inputfile> optional: -o <outputfile> -r <downsampling rate, e.g., 0.5> -g <log on>')
#         sys.exit(2)
#     for opt, arg in opts:
#         if opt in ['-h']:
#             print('strabo-util.py -i <inputfile> optional: -o <outputfile> -r <downsampling rate, e.g., 0.5> -g <log on>')
#             sys.exit()
#         elif opt in ['-i']:
#             inputfile = arg
#         elif opt in ['-o']:
#             outputfile = arg
#         elif opt in ['-r']:
#             resize_ratio = arg
#         elif opt in ['-g']:
#             log_on = arg
#     try:
#         all_process(inputfile, outputfile, resize_ratio, log_on)
#     except:
#         print('Something went wrong...')
def process_folder(folder_name):
    directory = os.fsencode(folder_name)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tif"):
            if not (filename.endswith("mask.tif")):
                all_process(imagery_fn=filename, directory=folder_name)
                print(filename + '\n')
                continue
        else:
            continue


if __name__ == "__main__":
    process_folder('/Users/yaoyic/Downloads/nls-maps/')
# main(sys.argv[1:])

# if __name__ == '__main__':
#
#     all_process('102343982.27-ori.tif')
#     all_process('102344992.27-ori.tif')
