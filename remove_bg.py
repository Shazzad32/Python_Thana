# from rembg import remove
# from PIL import Image
# inpurt_path = 'image_1.jpg'
# output_path = 'out.png'

# input = Image.open(inpurt_path)
# out = remove(input)
# out.save(output_path)

import numpy as np
from PIL import Image
import pymatting

# read the image and convert it to a numpy array
img = np.array(Image.open('image_1.jpg'))

# perform alpha matting to extract the foreground
alpha = pymatting.estimation.estimate_alpha_cf(
    img, np.ones(img.shape[:2]), method='cf', sigma=10**-6)

# create a new RGBA image with the extracted foreground
rgba = np.concatenate((img, alpha[..., np.newaxis]), axis=-1)

# save the output image
Image.fromarray((rgba * 255).astype(np.uint8)).save('output.png')
