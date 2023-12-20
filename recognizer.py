import easyocr
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Recognizer():

    def __init__(self, model_storage_directory="./easyocr_models", download_enabled=False):
        """generates objeects of class Recognizer

        Parameters
        ----------
        model_storage_directory : str, optional
            folder path where easyOCR models are stored, by default "."
        download_enabled : bool, optional
            If true easyOCR downloads latest models from repo, by default False
        """
        self.reader = easyocr.Reader(
            ['en'],
            model_storage_directory=model_storage_directory,
            download_enabled=download_enabled
            )

    @classmethod
    def read_image(self, image_path):
        """reads image file from path and returns cv2 image object

        Parameters
        ----------
        image_path : str
            Filepath to image

        Returns
        -------
        cv2.Image
            OpenCV image object
        """
        return cv2.imread(image_path)
    
    @classmethod
    def read_video_file(self, video_path):
        """Returns a video capture object of cv2

        Parameters
        ----------
        video_path : str
            Video filepath

        Returns
        -------
        cv2.VideoCapture
            Video capture object of cv2
        """
        return cv2.VideoCapture(video_path)

    def read_text_from_image(self, image):
        """reads text from image

        Parameters
        ----------
        image : cv2.Image
            OpenCv image object with text

        Returns
        -------
        results : dict
            Dict of recognized text, bounding box co-ordinates and probability
            
        image : cv2.image
            Image with bounding boxes
        """
        results = self.reader.readtext(image)

        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return results, image

    def read_text_from_image_file(self, image_filepath):
        """reads text from image provided as filepath

        Parameters
        ----------
        image_filepath : str
            Filepath to image with text

        Returns
        -------
        results : dict
            Dict of recognized text, bounding box co-ordinates and probability

        image : cv2.image
            Image with bounding boxes
        """
        return self.read_text_from_image(
            image=Recognizer.read_image(image_filepath)
        )

    def read_text_in_video_capture(self, video_capture):
        """reads text from video provided as filepath and dsiplays video with text

        Parameters
        ----------
        video_capture : cv2.VideoCapture
            Filepath to video with text

        Returns
        -------
        results : dict
            Dict of recognized text, bounding box co-ordinates and probability
        """
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Break the loop if the video has ended
            if not ret:
                break

            # Perform OCR on the frame
            results = self.reader.readtext(frame)

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

        return results
    
    def read_text_from_video_file(self, video_filepath):
        """reads text from image provided as filepath

        Parameters
        ----------
        video_filepath : str
            Filepath to video with text

        Returns
        -------
        results : dict
            Dict of recognized text, bounding box co-ordinates and probability

        """
        return self.read_text_in_video_capture(
            video_capture=Recognizer.read_video_file(video_filepath)
        )

if __name__=="__main__":

    recog = Recognizer()

    # Test: image
    image_path = 'input_files/image.jpg'
    results, image = recog.read_text_from_image_file(image_path)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Print the recognized text
    for (bbox, text, prob) in results:
        print(f"Text: {text}, Probability: {prob:.2f}")

    # Test: video
    # video_path = "input_files/video1.mp4"
    # results = recog.read_text_from_video_file(video_path)
    
    
    