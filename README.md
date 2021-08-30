# png_table_ocr
Description:
Uses Optical Character Recognition (OCR) to read a table from a 'png'

## Features
- Provides image preprocessing options to get most accurate OCR
- Has an auto scale method which helps pre-processing
- Creates chart lines through vertical and horizontal tolerances given by the user
- Reads the chart cell by cell to get the most accurate results
- Provides an option to create a csv with data processed from the OCR

## Setup

```bash
pip install numpy
pip install opencv-python
pip install pytesseract
```

## Configure PyTesseract

### On Linux 

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

### On Mac 

```bash
brew install tesseract
```

### On Windows 
download binary from https://github.com/UB-Mannheim/tesseract/wiki. then add
```bash
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
```
 to your script. (replace path of tesseract binary if necessary)
 
 ## Methods
 
 ```python
 def __init__(self, image_path: str, tesseract_path: str, show=False):
 ```
 Initializes the object
 image_path: str -> path to chart image you are trying to read
 tesseract_path: str -> On Windows you will need to give a path to tesseract.exe

 
 ```python
 def dilate_img(self) -> None:
 ```
 dilates the image using cv2.dilate w/ kernel = np.ones((3, 3), np.uint8)
 
  ```python
 def gaussian_blur(self) -> None:
 ```
 blurs the image using cv2.GaussianBlur
 
 ```python
 def grayscale(self) -> None:
 ```
 grayscales the image using cv2.cvtColor(<img path>, cv2.COLOR_BGR2GRAY)
 
 ```python
 def canny_img(self) -> None:
 ```
 alters the image using cv2.Canny
 
 ```python
 def morph_img(self) -> None:
 ```
 morphs the image using cv2.morphologyEx
 
 ```python
 def pre_process_image(self) -> None:
 ```
 diates, then blurs, then grayscales, then canny, then morphs image
 
 ```python
 def auto_resize_image(self) -> None:
 ```
Resizes the image so that either the height, width, or both are ~7000

 ```python
 def create_lines(self, rho=1, threshold=50, min_line_length=500, theta=(np.pi/180), max_line_gap=20) -> None:
 ```
 Defines the table lines with the above default values (values can be changed)
 
 ```python
 @staticmethod
 def __is_vertical(line, vert_tolerance: int) -> bool:
 ```
 returns whether the line is vertical or not based on vertical tolerance (how far x1 and x2 can be apart to still be considered vertical); default value of 100
 
 ```python
 @staticmethod
 def __is_horizontal(line, horizontal_tolerance: int) -> bool:
 ```
 returns whether the line is horizontal or not based on horizontal tolerance (how far y1 and y2 can be apart to still be considered vertical); default value of 100
 
 ```python
 def __overlapping_filter(self, lines: list, sorting_index: int, overlapping_tolerance=None) -> list:
 ```
 Combines overlapping lines based on a tolerance given; default overlapping tolerance = 50
 
 ```python
 ```
  
 ```python
 ```
  
 ```python
 ```
  
 ```python
 ```
 
 ## Example Usage
 
```python
from png_table_ocr import ReadTableImage

filename = 'test_1.png'

img = ReadTableImage(image_path=filename, tesseract_path=(r'C:\Users\JSPANGLER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'), show=False)
img.auto_resize_image()
img.pre_process_image()
img.create_merge_lines(vert_tolerance=100, horizontal_tolerance=50, overlap_tolerance=100)
chart_data = img.read_chart()
for line in chart_data:
    print(line)

```
