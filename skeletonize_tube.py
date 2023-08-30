import skimage, os
from skimage import io, filters, morphology, draw

import matplotlib.pyplot as plt
import numpy as np

def skel_chromosomes(img):
    redchannel = img[:,:,0]

    tubey = skimage.filters.sato(redchannel,
                                 sigmas = range(1,2,1),
                                 black_ridges = False)
    
    thresh = skimage.filters.threshold_otsu(tubey)
    binary = tubey > (thresh * 0.7)

    big_clean = skimage.morphology.remove_small_objects(binary, 100)
    
    label_image = skimage.measure.label(big_clean)
    props = skimage.measure.regionprops(label_image)
    
    areas = [obj.area for obj in props]
    largest = areas.index(max(areas))

    largest_centroid = np.array(props[largest].centroid)
    largest_axis = props[largest].axis_major_length
    
    centroids = [np.array(obj.centroid) for obj in props]
    distances = [np.linalg.norm(centroid - largest_centroid) for
                 centroid in centroids]
    
    remove_idx = [region+1 for region,distance in enumerate(distances)
                  if distance > largest_axis]
    
    if remove_idx:
        remove_regions = [label_image == idx for idx in remove_idx]
        remove_regions = np.logical_or.reduce(remove_regions)
        big_clean = np.logical_and(big_clean,
                                   np.logical_not(remove_regions))
    
    hull = skimage.morphology.convex_hull_image(big_clean)
    little_clean = skimage.morphology.remove_small_objects(binary, 20)
    
    cleaned = np.logical_and(little_clean, hull)
    skeleton = skimage.morphology.skeletonize(cleaned)
    
    return cleaned, skeleton

def skel_plot(path):
    img = io.imread(path)

    cleaned, skeleton = skel_chromosomes(img)

    overlay = img.copy()
    overlay[:,:,1] = overlay[:,:,2] = skeleton * 255

    f, axarr = plt.subplots(2,2, figsize = (10,8))
    
    axarr[0,0].imshow(img)
    axarr[0,0].axis('off')
        
    axarr[0,1].imshow(cleaned)
    axarr[0,1].axis('off')
        
    axarr[1,0].imshow(skeleton)
    axarr[1,0].axis('off')
        
    axarr[1,1].imshow(overlay)
    axarr[1,1].axis('off')
        
    plt.suptitle(path.split('/')[-1])
