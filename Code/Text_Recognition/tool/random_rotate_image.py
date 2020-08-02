from PIL import Image
import random
import os

inputPath = "../data/Latin_images/"
outputPath = "../data/Latin_images/Rotated/"

for image_file_name in os.listdir(inputPath):
    if image_file_name.endswith(".jpg"):
        img = Image.open(inputPath+image_file_name)
        rotate = random.randint(-10, 10)
        output = img.rotate(rotate, expand=True)
        output.save(outputPath + image_file_name)