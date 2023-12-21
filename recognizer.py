import easyocr
import matplotlib.pyplot as plt
import cv2 as cv
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
    def read_image(self, image_path, grayscale=False):
        """reads image file from path and returns cv image object

        Parameters
        ----------
        image_path : str
            Filepath to image

        Returns
        -------
        cv.Image
            OpenCV image object
        """
        if grayscale:
            return cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        else:
            return cv.imread(image_path)
    
    @classmethod
    def read_video_file(self, video_path):
        """Returns a video capture object of cv

        Parameters
        ----------
        video_path : str
            Video filepath

        Returns
        -------
        cv.VideoCapture
            Video capture object of cv
        """
        return cv.VideoCapture(video_path)

    def read_text_from_image(self, image):
        """reads text from image

        Parameters
        ----------
        image : cv.Image
            OpenCv image object with text

        Returns
        -------
        results : dict
            Dict of recognized text, bounding box co-ordinates and probability
            
        image : cv.image
            Image with bounding boxes
        """
        results = self.reader.readtext(image)

        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv.putText(image, text, top_left, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

        image : cv.image
            Image with bounding boxes
        """
        return self.read_text_from_image(
            image=Recognizer.read_image(image_filepath)
        )

    def read_text_in_video_capture(self, video_capture):
        """reads text from video provided as filepath and dsiplays video with text

        Parameters
        ----------
        video_capture : cv.VideoCapture
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
                    cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    cv.putText(frame, text, top_left, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Display the resulting frame
            cv.imshow('Video', frame)

            # Break the loop if 'q' key is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all windows
        video_capture.release()
        cv.destroyAllWindows()

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
        
    def shape_detection(self, image, template, method="cv.TM_CCOEFF"):
        img = image.copy()
        w, h = template.shape[::-1]
        result = cv.matchTemplate(img, template, eval(method))
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        
        return result, img

if __name__=="__main__":

    recog = Recognizer()

    # Test: image
    # image_path = 'input_files/image.jpg'
    # results, image = recog.read_text_from_image_file(image_path)

    # plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # # Print the recognized text
    # for (bbox, text, prob) in results:
    #     print(f"Text: {text}, Probability: {prob:.2f}")

    # Test: video
    # video_path = "input_files/video1.mp4"
    # results = recog.read_text_from_video_file(video_path)
    
    
    # Test: Template matching
    image_path = 'input_files/image.jpg'
    template_path = 'input_files/arrow_temp.jpg'
    
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    for meth in methods:
        method = meth
        # Apply template Matching
        result, image = recog.shape_detection(
            image=Recognizer.read_image(image_path, grayscale=True),
            template=Recognizer.read_image(template_path, grayscale=True),
            method=method
        )

        plt.subplot(121),plt.imshow(result, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
    
    