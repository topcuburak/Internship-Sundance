  # HISTOGRAM EQUALIZATION
  
  Histogram equalization is an algorithm used to manipulate pixel densities in grayscale images to increase understandilibity of the images. 
For more detail about the algorithm, one can visit corresponding wikipedia page from https://en.wikipedia.org/wiki/Histogram_equalization. Also, shared pdf file is the presentation that was presented in the HiPEAC student challenge evet. This file also describes the algorithm.

  In this work, we tried to compare power consumption and performance values among CPU, GPU and FPGA. The source code of the algorithm implemented in Verilog can be found in the src files. 
  
  For the ones who want to test the algorithm:
  1) Choose a png and generate the corresponding hexadecimal pixel values in a txt files by using png_to_text_converter.py file.\
     -python3 png_to_text_converter.py file_name.png hex_output.txt
  2) Run the source code with test bench. 
     -In the testbench code, please update the path of the hex_output.txt file. Also, update the width and height parameters in both source code and testbench.   
  3) Run your code and generate the resultant .txt file. 
  4) Re-run your python code to convert this resultant .txt file to the .png file.\
     -python3 png_to_text_converter.py resultant.txt resultant.png original.png

  In this way, one can try the algorithm and observe the results.
  This algorithm consumes 2.047 watt/sec power and 2.047 joule/sec for an image whose width and height dimensions are equal to 512.  Also, algorithm results in 10.49 msec to manipulate the same image. However, reading and re-writing the image from/to memory times are not included in this measurement. \
  
  Explanation for the data: \
i) PNG files:
    - im.png (256x256), 
    - lenna.png (512x512), 
    - lyon_4096.png (4096x4096) \
ii) Hexadecimal conversions of the original PNG files:
    - im_hex.txt
    - lenna_hex.txt
    - lyon_hex.txt \
iii) Hexadecimal results after histogram equalization is applied:
    - im_res.txt
    - lenna_res.txt
    - lyon_res.txt (AWS EC2 server) \
iv) Resultant PNG files after histogram equalization is applied:
    - im_res.png
    - lenna_res.png
    - lyon_res.png (AWS EC2 server) \
  
  Since Lyon image is bigger in terms of file size, I could not synthesize/implement this image in my local. Hence, I have used the AWS EC2 server to test it. 

  The mentioned results are obtained from the VCS-1 board provided by the Sundance Multiprocessor tech.  (https://www.sundance.com/vcs-1/) \
  Power measuremetns are done with the help of LynSyn Lite device.  (https://store.sundance.com/product/lynsyn/ , https://www.xilinx.com/products/boards-and-kits/1-18tmbgt.html)
  																	 
 
![Screenshot from 2021-10-31 17-27-43](https://user-images.githubusercontent.com/73446582/139588356-15aa5e86-5c13-453f-9142-923f9bb0d6aa.png)
 
  
![Screenshot from 2021-10-31 17-28-25](https://user-images.githubusercontent.com/73446582/139588382-1616353d-abb2-43e1-a404-81089cb1efde.png)
