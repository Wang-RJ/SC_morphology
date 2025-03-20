import skimage, os, sys, argparse

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

def process_image(input_path, output_csv, overlay_folder):
    with open(output_csv, 'a') as outfile:
        image = skimage.io.imread(input_path)    
        cleaned, skeleton = skel_chromosomes(image)
        
        clean_size = sum(sum(cleaned))
        skel_size  = sum(sum(skeleton))
        
        outfile.write(input_path + ", " +
                      str(clean_size) + ", " +
                      str(skel_size) + "\n")
        
        if overlay_folder:
            overlay = image.copy()
            overlay[:,:,1] = overlay[:,:,2] = skeleton * 255
        
            filename = input_path.split("/")[-1]
            outpath = os.path.join(overlay_folder,
                                   filename.split(".")[0] + "_overlay.jpg")

            skimage.io.imsave(outpath, overlay, check_contrast = False)

def main():
    parser = argparse.ArgumentParser(description="Get skeleton size for MLH1 image")

    # Adding required arguments
    parser.add_argument("input_path", type=str,
                        help="the path to the input cytological image")
    parser.add_argument("output_csv", type=str,
                        help="the path to the output .csv tallying skeleton size")
    
    parser.add_argument("--overlay", type=str, dest="OVERLAY_FOLDER",
                        help="(optional) overlay image saved to OVERLAY_FOLDER",
                        default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: The provided input path does not exist: {args.input_path}", file=sys.stderr)
        sys.exit(1)
    if args.overlay_folder:
        os.makedirs(args.overlay_folder, exist_ok=True)
        
    process_image(args.input_path, args.output_csv, args.overlay_folder)

if __name__ == '__main__':
    main()