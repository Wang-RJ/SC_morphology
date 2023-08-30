import skimage, os
from skimage import io, filters, morphology, draw

import matplotlib.pyplot as plt
import numpy as np

def skel_chromosomes(img, path = None, do_plot = False):
    redchannel = img[:,:,0]

    rc = skimage.exposure.equalize_hist(redchannel)
    binary = rc > (rc.max() * 0.97)

    cleaned = skimage.morphology.remove_small_objects(binary, 150)
    skeleton = skimage.morphology.skeletonize(cleaned)
    
    clean_size = sum(sum(cleaned))
    skel_size = sum(sum(skeleton))

    if(do_plot):
        f, axarr = plt.subplots(2,2, figsize = (10,8))
        axarr[0,0].imshow(img)
        axarr[0,1].imshow(binary)
        axarr[1,0].imshow(cleaned)
        axarr[1,1].imshow(skeleton)
        plt.suptitle(path.split('/')[-1])
    
    return clean_size, skel_size
