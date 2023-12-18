import easyocr
import matplotlib.pyplot as plt
import cv2
import re

# Replace 'en' with the language code you want to use (e.g., 'en' for English)
reader = easyocr.Reader(['en'])

# Replace 'your_image_path.jpg' with the path to your image file
image_path = 'input_files/image.jpg'

image = cv2.imread(image_path)

# Use EasyOCR to read the text from the image
results = reader.readtext(image)

# Replace all the o's in text to zero
# corrected_text = re.sub(r'\bo\b', '0', results[1], flags=re.IGNORECASE)
# print(corrected_text)
#
# Display the image with bounding boxes and recognized text
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print the recognized text
for (bbox, text, prob) in results:
    print(f"Text: {text}, Probability: {prob:.2f}")
