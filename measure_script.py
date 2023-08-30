import skeletonize_tube

absolute_path = "~/synaptonemal/"
out_path = absolute_path + "SC_measures.csv"

folders = [200, 201, 202, 203, 204, 208, 212, 217, 233, 234]
folders = [absolute_path + str(x) + '/' for x in folders]

allpaths = [[path + file for file in os.listdir(path)] for path in folders]

flatpaths = [flatpath for folder in allpaths for flatpath in folder]
flatnames = [path.split('/')[-1] for path in flatpaths]

for path,name in zip(flatpaths, flatnames):
    with open(out_path, 'a') as outfile:
        image = io.imread(path)
    
        cleaned, skeleton = skel_chromosomes(image)
        
        clean_size = sum(sum(cleaned))
        skel_size  = sum(sum(skeleton))
        
        outfile.write(name + "," +
                      str(clean_size) + "," +
                      str(skel_size) + "\n")
        
        overlay = image.copy()
        overlay[:,:,1] = overlay[:,:,2] = skeleton * 255
        
        folder = name.split('-')[0]
        
        skimage.io.imsave("overlays/" + folder + "/" +
                          name.split(".")[0] + "_overlay.jpg",
                          overlay,
                          check_contrast = False)
