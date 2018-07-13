# Crop all images in a dataset 
# Assuming that background is transparent/white
# Look for min/max locations of colour pixels
# Creates a new dir from the original olddir called olddir_cropped

import cv2 as cv
import os
import sys
import numpy as np

def segment_background(input_image):
  # Boundary for red (white looks red in hsv)
  boundaries = [[30,150,50], [255,255,180]]

  # Upper and lower colour boundaries
  lower = np.array(boundaries[0], dtype = "uint8")
  upper = np.array(boundaries[1], dtype = "uint8")

  # Convert to hsv
  hsv = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)

  # Mask out the red
  mask = cv.inRange(hsv, lower, upper)
  output = cv.bitwise_and(input_image, input_image, mask = mask)

  # Convert to greyscale
  gray_image = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

  # Do an inverse thresholde - product left in white
  ret,thresh1 = cv.threshold(gray_image,0,255,cv.THRESH_BINARY)

  # Convert back to colour
  # colour_image = cv.cvtColor(thresh1, cv.COLOR_GRAY2BGR)

  # cv.imshow( "thresh1", thresh1 )
  # cv.waitKey( 100 )
  # cv.imshow( "gray", gray_image )
  # cv.waitKey( 10 )
  # cv.imshow( "render1_imp", output )
  # cv.waitKey( 10 )
  # cv.imshow( "render1_imp2", hsv )
  # cv.waitKey( 0 )

  return thresh1

def cycle_images(org_dir, cropped_dir, perf_background):
  # Extract all the images
  if not os.path.isdir(org_dir):
    return
  print("Dir: {}" .format(org_dir))
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

      if perf_background == 0:
        thresh_image = segment_background(image)
      else:
        thresh_image = image

      cv.imshow( "thresh1", thresh_image )
      cv.waitKey( 100 )
      # Find the shape
      rows = thresh_image.shape[0]
      cols = thresh_image.shape[1]

      # Initialise max and mins
      min_col = cols 
      max_col = 0
      min_row = rows
      max_row = 0

      locations = cv.findNonZero(thresh_image)

      # Find the bounding boxes
      for i in range(0, len(locations)):
        row = locations[i,0][1]
        col = locations[i,0][0]
        # print(locations[i,0][0])
      # for (int i = 0; i < locations.total(); i++ ) {
      #   // Extract the point
      #   int row = locations.at<cv::Point>(i).y;
      #   int col = locations.at<cv::Point>(i).x;

        if(row > max_row):
          max_row = row
        if(row < min_row):
          min_row = row
        if(col > max_col):
          max_col = col
        if(col < min_col):
          min_col = col
      # }

      # Cycle through the pixels
      # for idx, row in enumerate(thresh_image):
      #   for idy, pixel in enumerate(row):
      #     # Segregate the coloured pixels
      #     # if pixel[0] == 0:
      #     #   print(pixel[0], pixel[1], pixel[2])

      #     if pixel[0] != 255 and pixel[1] != 255 and  pixel[2] != 255:
      #       row = idx
      #       col = idy

      #       # Find if there's new max and min values
      #       if(row > max_row):
      #         max_row = row
      #       if(row < min_row):
      #         min_row = row
      #       if(col > max_col):
      #         max_col = col
      #       if(col < min_col):
      #         min_col = col

      print(min_row, max_row, min_col, max_col)
      # Select the desired shape
      cropped = image[min_row:max_row, min_col:max_col]

      # Save the image
      cv.imwrite(cropped_path, cropped)

def main(parent_dir, cropped_dirs, perf_background):
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
      cycle_images(dirs[idx], cropped_dirs_path, perf_background)
  # Cycle through the dirs

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("usage: python crop_products.py <foldername>")
    exit()
  else:
    # Define the dir where the original products are
    PARENT_DIR = sys.argv[1]
  # Whether background is perfectly white or not
  perf_background = 0

  # Create the new directory where the cropped images will be saved
  # if last char is a slash
  if PARENT_DIR[-1] == "/":
    CROPPED_DIRS = PARENT_DIR[:-1] + "_cropped/"
  else :
    CROPPED_DIRS = PARENT_DIR + "_cropped/"

  print("New dir {} " .format(CROPPED_DIRS))
  main(PARENT_DIR, CROPPED_DIRS, perf_background)