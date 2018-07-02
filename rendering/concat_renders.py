# Concat images and produce an output
# Downsamples the images to half their width

import cv2 as cv
import glob
import numpy as np
import os
import shutil
from skimage import transform as tf

# Define the two folders to read the renders from
# Assuming the same number of renders in each
folder_path_1 = "input_path_1"
folder_path_2 = "input_path_2"
output_folder = "output_path"
folder_paths = [folder_path_1, folder_path_2]

# Add text to an image
def add_text(image, text, loc):
  # Define parameners
  shape = image.shape
  font                   = cv.FONT_HERSHEY_SIMPLEX
  # Loc == 0 - bottom of the image 
  # Loc == 1 - top left 
  # Loc == 2 - top more left 
  if loc == 0:
    bottomLeftCornerOfText = (int(shape[1]/2),500)
  elif loc == 1 :
    bottomLeftCornerOfText = (int(shape[1]/3), 40)
  elif loc == 2 :
    bottomLeftCornerOfText = (20, 30)

  fontScale              = 1
  fontColor              = (255,255,255)
  lineType               = 2

  # Add the text to the image
  cv.putText(image, text, 
      (bottomLeftCornerOfText), 
      font, 
      fontScale,
      fontColor,
      lineType)

def main():

  # Create the output folder if it doesn't already exist
  if os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
  os.makedirs(output_folder)

  # Load the filenames
  filenames_1 = [img for img in glob.glob(folder_path_1 + "/*.png")]
  filenames_2 = [img for img in glob.glob(folder_path_2 + "/*.png")]

  # Order the filenames
  filenames_1.sort() 
  filenames_2.sort() 

  # Put into a list
  files = [filenames_1, filenames_2]

  # Cycle through the lists of files
  for img in range(0,len(filenames_1)):
    print("File {} " .format(img))

    # Load in the file names
    file_paths = [files[0][img], files[1][img]]

    # Read in the images
    images = [cv.imread(file_paths[0]), cv.imread(file_paths[1])]

    # Find the shape of the images
    shape = images[0].shape

    # Add text to the images
    add_text(images[0], "Image_1", 2)
    add_text(images[1], "Image_2", 2)

    # Halve the width
    im_1_r = cv.resize(images[0], dsize=(int(shape[1]/2), int(shape[0])), interpolation=cv.INTER_CUBIC)
    im_2_r = cv.resize(images[1], dsize=(int(shape[1]/2), int(shape[0])), interpolation=cv.INTER_CUBIC)

    # Concatenate the images
    vis = np.concatenate((im_1_r, im_2_r), axis=1)

    # Create the name of the file so it fits the convention for creating a video
    # Must be 0005 etc
    file_name = "img_"
    # Desired name length
    img_name_len = 5
    if len(str(img)) < img_name_len:
      # Find the difference between the desired length and cur length
      difference = img_name_len - len(str(img))
      img_number = str(0)*difference
      img_number = img_number + str(img)
    else:
      img_number = str(img)

    # Make the last image the first image so it shows at the end of the video
    if img == len(filenames_1) - 1:
      img_number = str(0)*img_name_len
      print("Image number: {}" .format(img_number))
    # Create the file name
    file_name = file_name + str(img_number)

    # Save the image
    cv.imwrite( output_folder + str(file_name) + '.png', vis)

if __name__ == "__main__":
  main()