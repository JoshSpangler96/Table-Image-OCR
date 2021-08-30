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
 
 ## Example Usage
 
```python
from png_table_ocr import ReadTableImage

filename = 'test_3.png'

img = ReadChartImage(image_path=filename, tesseract_path=(r'C:\Users\JSPANGLER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'), show=False)
img.auto_resize_image()
img.pre_process_image()
img.create_merge_lines(vert_tolerance=100, horizontal_tolerance=50, overlap_tolerance=100)
chart_data = img.read_chart()
for line in chart_data:
    print(line)

```
