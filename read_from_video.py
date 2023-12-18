import cv2
import easyocr

# Set the language for OCR (e.g., 'en' for English)
language = 'en'

# Create an EasyOCR reader
reader = easyocr.Reader([language])

# Path to the video file
video_path = 'input_files/video1.mp4'

# Open the video file
video_capture = cv2.VideoCapture(video_path)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Perform OCR on the frame
    results = reader.readtext(frame)

    # Display the results
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        if prob > 0.5 and top_right[0] - top_left[0] > 300:
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw bounding box and text on the frame
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
