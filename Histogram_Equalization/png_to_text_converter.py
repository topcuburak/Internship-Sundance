from PIL import Image as image
import numpy as np
import getopt, sys # for taking argument from the command line 

# to convert png image to the hexadecimal format that is readable by verilog
def png_to_text_converter(png_name, text_output_name):

    img = image.open(png_name).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata()) # convert image data to a list of integers

    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

    f = open(text_output_name, 'w')

    for i in range (0, HEIGHT):
        for j in range (0, WIDTH):
            if data[i][j] <= 15:
                conv = ""
                for c in hex(data[i][j]):
                    if c != 'x':
                        conv += c
                    else:
                        conv += 'x0'
                f.write(conv)
                f.write('\n')
            else:
                f.write(hex(data[i][j]))
                f.write('\n')


# to convert hexadecimal image to the png format benefitting from the original image's header file
def text_to_png_converter(text_file_name, png_output_name, original_png):
    img = image.open(original_png).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    result = open(text_file_name, 'r').readlines()

    for i in range (0, WIDTH*HEIGHT):
        if 'xx' in result[i]:
            img.putpixel((int(i%WIDTH),int(i/HEIGHT)), (255))
        else:
            img.putpixel((int(i%WIDTH),int(i/HEIGHT)), (int(result[i].rstrip('\n'), 16)))


    img.save(png_output_name)


# example image data including 8*8 pixels each have 8bits of depth
def example():
    arr = [[52, 55, 61, 59, 70, 61, 76, 61],
           [62, 59, 55, 104, 94, 85, 59, 71],
           [63, 65, 66, 113, 144, 104, 63, 72],
           [64, 70, 70, 126, 154, 109, 71, 69],
           [67, 73, 68, 106, 122, 88, 68, 68],
           [68, 79, 60, 79, 77, 66, 58, 75],
           [69, 85, 64, 58, 55, 61, 65, 83],
           [70, 87, 69, 68, 65, 73, 78, 90]]

    f = open('test_arr.txt', 'w')

    for i in range (0, len(arr)):
        for j in range (0, len(arr[0])):
            f.write(hex(arr[i][j]))
            f.write('\n')


argumentList = sys.argv[1:]

if (len(argumentList) == 3):
    text_to_png_converter(argumentList[0], argumentList[1], argumentList[2])

elif (len(argumentList) == 2):
    png_to_text_converter(argumentList[0], argumentList[1])

else:
    example()
