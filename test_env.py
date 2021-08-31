from table_img_ocr import ReadTableImage

#image path
filename = 'test_3.png'

img = ReadTableImage(image_path=filename, tesseract_path=(r'C:\Users\JSPANGLER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'), show=False)
img.auto_resize_image()
img.pre_process_image()
img.create_merge_lines(vert_tolerance=100, horizontal_tolerance=50, overlap_tolerance=100)
chart_data = img.read_chart()
img.create_csv()
for line in chart_data:
    print(line)
