from table_img_ocr import ReadTableImage
import pdf_to_image

#image path
filename = 'example_1.png'

img = ReadTableImage(image_path=filename, tesseract_path=(r'C:\Users\JSPANGLER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'), show=False)
img.auto_resize_image()
img.pre_process_image()
img.create_merge_lines()
chart_data = img.read_chart()
img.create_csv()
for line in chart_data:
    for i in range(len(line)):
        line[i] = line[i].replace('|', ' ')
    print(line)
