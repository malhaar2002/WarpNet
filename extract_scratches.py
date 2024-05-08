import cv2
import numpy as np

def remove_background_and_extract_scratches(input_image_path, output_scratches_path):
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate the scratches from the background
    _, binary_image = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)

    # Find contours of the scratches
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to extract the scratches
    scratch_mask = np.zeros_like(grayscale_image)

    # Draw contours on the mask
    cv2.drawContours(scratch_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image to extract the scratches
    extracted_scratches = cv2.bitwise_and(input_image, input_image, mask=scratch_mask)

    # Save the extracted scratches as an image
    cv2.imwrite(output_scratches_path, extracted_scratches)

    print("Scratches extracted and saved successfully!")

# Example usage
input_image_path = "images/old_image_textures/folded_1.jpg"
output_scratches_path = "extracted_scratches.jpg"  # Path to save the extracted scratches
remove_background_and_extract_scratches(input_image_path, output_scratches_path)
