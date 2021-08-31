from read_chart_ocr import ReadChartImage

#image path
filename = 'test_3.png'

img = ReadChartImage(image_path=filename, tesseract_path=(r'C:\Users\JSPANGLER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'), show=False)
img.auto_resize_image()
img.pre_process_image()
img.create_merge_lines(vert_tolerance=100, horizontal_tolerance=50, overlap_tolerance=100)
chart_data = img.read_chart()
for line in chart_data:
    print(line)
