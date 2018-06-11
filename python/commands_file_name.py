# Save/move files based on their name
# File format: render_123.png
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
    main()