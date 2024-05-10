import os
import cv2
import random
from PIL import Image
from unstructured_noise import online_add_degradation

input_path = r"D:\WarpNET\coco_persons\B\test"
output_path = r"D:\WarpNET\coco_persons\A\test"
filters_path = r"D:\WarpNET\WarpNet\images\old_image_textures"



# iterate through all images in the input path
for image_path in os.listdir(input_path):
    image = cv2.imread(os.path.join(input_path, image_path))
    image = cv2.resize(image, (256, 256))
    filter = cv2.imread(os.path.join(filters_path, random.choice(os.listdir(filters_path))))
    filter = cv2.resize(filter, (256, 256))
    image = cv2.addWeighted(image, 0.5, filter, 0.5, 0)
    cv2.imwrite(os.path.join(output_path, image_path), image)
    # else:
    #     # unstructured defect
    #     image = online_add_degradation(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    #     image.save(os.path.join(output_path, image_path))