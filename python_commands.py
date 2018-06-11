# 1. Cut off last work of a sentence
sentence = "Hello I am here"
# Use rsplit() - reverse split
# , 1 - make a single split
# ' ' - split on spaces
# [0] - take first
sentence = sentence.rsplit(' ', 1)[0]


# 2. List contents of a folder
import os
os.listdir("path") # returns list

# 3. Add text to a file
# Add text to an image
def add_text(image, text, loc):
  # Define parameners
  shape = image.shape
  font                   = cv.FONT_HERSHEY_SIMPLEX
  # Loc == 0 - bottom of the image 
  # Loc == 1 - bottom of the image 
  if loc == 0:
    text_loc = (int(shape[1]/2),500)
  elif loc == 1 :
    text_loc = (int(shape[1]/3), 40)
  elif loc == 2 :
    text_loc = (20, 30)

  fontScale              = 1
  fontColor              = (255,255,255)
  lineType               = 2

  # Add the text to the image
  cv.putText(image, text, (text_loc), font, fontScale,
      fontColor, lineType)

# Save/move files based on their name
# File format: render_123.png
