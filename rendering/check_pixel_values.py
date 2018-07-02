import cv2 as cv
import os
import numpy as np 

folder_path = "/home/buckleyl/Documents/Code/Repos/MvBot/DeepLearning/apps/similarity_measure/datasets/training/render1/test/"
def main():

  files = os.listdir(folder_path)

  colours = []
  for idx, img in enumerate(files):
    print(idx)
    path = folder_path + img
    image = cv.imread(path)
    shape = image.shape

    avg_colour_per_row = np.average(image, axis = 0)
    average_colour = np.average(avg_colour_per_row, axis=0)

    print(average_colour)

    if any(x == average_colour[0] for x in colours):
      print("Already present!\n\n")
    colours.append(average_colour[0])


    cv.imshow( "render1_gen", image )
    cv.waitKey( 100 )

if __name__ == "__main__":
  main()