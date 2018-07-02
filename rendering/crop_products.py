# Crop all images in a dataset 
# Assuming that background is transparent
# Look for min/max locations of colour pixels
# Creates a new dir from the original olddir called olddir_cropped

import cv2 as cv
import os

def cycle_images(org_dir, cropped_dir):
  # Extract all the images
  image_paths = os.listdir(org_dir)

  print(image_paths)
  for img in image_paths:
    # Create the full paths
    cropped_path = cropped_dir + "/" + img
    image_path = org_dir + "/" + img

    # Load the image
    # print("Loading {}" .format(image_path))
    if 'png' in image_path:
      image = cv.imread(image_path)

      # Find the shape
      rows = image.shape[0]
      cols = image.shape[1]

      # Initialise max and mins
      min_col = cols 
      max_col = 0
      min_row = rows
      max_row = 0

      # Cycle through the pixels
      for idx, row in enumerate(image):
        for idy, pixel in enumerate(row):
          # Segregate the coloured pixels
          if pixel[0] != pixel[1] != pixel[2] != 255:
            row = idx
            col = idy

            # Find if there's new max and min values
            if(row > max_row):
              max_row = row
            if(row < min_row):
              min_row = row
            if(col > max_col):
              max_col = col
            if(col < min_col):
              min_col = col

      # Select the desired shape
      cropped = image[min_row:max_row, min_col:max_col]

      # Save the image
      cv.imwrite(cropped_path, cropped)

def main(parent_dir, cropped_dirs):
  # Find list of all folders in home dir
  dirs = os.listdir(parent_dir)
  full_cropped_dirs = []
  # Add the parent dir to the path
  # Enumerate: https://docs.python.org/3/library/functions.html#enumerate
  for idx, dir in enumerate(dirs):
    # Make sure it's not README.txt
    if 'txt' not in dir:
      # Create new dir path
      cropped_dirs_path = cropped_dirs + dir
      # Append to the list of new dirs
      full_cropped_dirs.append(cropped_dirs_path)

      # Create the dir if it doesn't already exist
      if not os.path.exists(cropped_dirs_path):
        os.makedirs(cropped_dirs_path)

      # Create full path of original dir
      dir = parent_dir + dir
      dirs[idx] = dir
      print("{}" .format(dirs[idx]))
      cycle_images(dirs[idx], cropped_dirs_path)
  # Cycle through the dirs

if __name__ == "__main__":

  # Define the dir where the original products are
  PARENT_DIR = "/home/buckleyl/Documents/Models/REFILLS/Baseline_Dataset_Extended/"

  # Create the new directory where the cropped images will be saved
  # if last char is a slash
  if PARENT_DIR[-1] == "/":
    CROPPED_DIRS = PARENT_DIR[:-1] + "_cropped/"
  else :
    CROPPED_DIRS = PARENT_DIR + "_cropped/"

  print("New dir {} " .format(CROPPED_DIRS))
  main(PARENT_DIR, CROPPED_DIRS)