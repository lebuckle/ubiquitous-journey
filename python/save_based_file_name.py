# Save/move files based on their name
# File format: render_123.png
import cv2 as cv
import os

original_file_path = "/home/buckleyl/Documents/Models/DECO_video/XOR_renders/liffey_3_build/"
target_file_path = "/home/buckleyl/Documents/Models/DECO_video/XOR_renders/liffey_3_build_divide/"

def main(original_path, target_path):

	# list the files
	renders = os.listdir(original_path)

	# Split the render names
	split_name = []
	for idx, render in enumerate(renders):
		render_name = render.split('_')[1]
		render_name = render_name.split('.')[0]
		# Convert to int
		render_name = int(render_name)

		# check if it meets the specification
		if render_name % 10 == 0:
			render_path = target_path + render
			# read in original render
			image_path = original_path + render
			image = cv.imread(image_path)
			cv.imwrite(render_path, image)
			print(render_path)

if __name__ == "__main__":
  main(original_file_path, target_file_path)