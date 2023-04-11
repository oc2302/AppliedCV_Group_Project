import histomicstk as htk
from PIL import Image
import large_image
import openslide
import numpy as np
import matplotlib.pyplot as plt
import openslide.deepzoom

# #Reads one slide
# source = openslide.OpenSlide("1-sample.ndpi")

# thumbnail = source.get_thumbnail((1024, 1024))
# #plt.imshow(thumbnail)
# #plt.show()


ts = large_image.getTileSource("1-sample.ndpi")

num_tiles = 0

tile_means = []
tile_areas = []

for tile_info in ts.tileIterator(
    region=dict(left=5000, top=5000, width=20000, height=20000, units='base_pixels'),
    scale=dict(magnification=20),
    tile_size=dict(width=1000, height=1000),
    tile_overlap=dict(x=50, y=50),
    format=large_image.tilesource.TILE_FORMAT_PIL
):

    im_tile = np.array(tile_info['tile'])
    tile_mean_rgb = np.mean(im_tile[:, :, :3], axis=(0, 1))

    tile_means.append( tile_mean_rgb )
    tile_areas.append( tile_info['width'] * tile_info['height'] )

    num_tiles += 1

slide_mean_rgb = np.average(tile_means, axis=0, weights=tile_areas)


#Sample Calculations for slide
print('Number of tiles = {}'.format(num_tiles))
print('Slide mean color = {}'.format(slide_mean_rgb))


#Show Thumbnail
im_low_res, _ = ts.getRegion(
    scale=dict(magnification=1.25),
    format=large_image.tilesource.TILE_FORMAT_NUMPY
)

plt.imshow(im_low_res)