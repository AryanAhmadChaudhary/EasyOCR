import cv2
import easyocr
import matplotlib.pyplot as plt

class EasyOCRDemo:
    def __init__(self, image_path, language='en', gpu=False):
        self.image_path = image_path
        self.reader = easyocr.Reader([language], gpu=gpu)
        self.extracted_texts = []
    
    def read_image(self):
        self.image = cv2.imread(self.image_path)
    
    def detect_text(self):
        """Detects text in the image and stores the bounding boxes and texts."""
        self.result = self.reader.readtext(self.image_path)
        self.extracted_texts = [detection[1] for detection in self.result]
    
    def annotate_image(self):
        """Annotates the image with bounding boxes and detected text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection in self.result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]

            """Draw rectangle and put text"""
            self.image = cv2.rectangle(self.image, top_left, bottom_right, (0, 255, 0), 3)
            self.image = cv2.putText(self.image, text, top_left, font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    def display_image(self):
        """Displays the annotated image."""
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    def print_extracted_texts(self):
        """Prints the list of all detected texts."""
        print("Extracted Texts:")
        for text in self.extracted_texts:
            print(text)

    def run(self):
        self.read_image()
        self.detect_text()
        self.annotate_image()
        self.display_image()
        self.print_extracted_texts()

# Usage
demo = EasyOCRDemo(image_path="path to your image")
demo.run()
