`timescale 1ns / 1ps

module histogram_equalizer(
    input [7:0] pixel_val,
    output reg [7:0] out_pixel_val,
    output reg finish,
    input clk,
    input ready,
    output reg [7:0] current_pixel,
    input read_start,
    output reg [12:0] y_counter,
    output reg [12:0] x_counter,
    output reg [3:0] step_counter,
    output reg [7:0] cdf_min,
    output reg min_found        
    ); 


// one can change those parameters with respect to the image size and pixel depth for grayscale images. 
    parameter width = 4096, 
              height = 4096,
              max_pixel_val = 256;

reg [24:0] gray_level [0:max_pixel_val-1];
reg [24:0] cdf [0:max_pixel_val-1];
reg [7:0] image [0:height-1][0:width-1];
reg [7:0] last_cdf_idx;

integer i;

//initialization step
initial
begin
    last_cdf_idx <= 0;
    //index counters of image
    y_counter <= 0;
    x_counter <= 0;
    //step counter of the algorithm
    step_counter <= 0;
    cdf_min <= 0;
    min_found <= 0;
    finish <= 0;
    out_pixel_val <= 0;
    //gray level holds the histogram valeus (occurrencies) of grayscale image pixels
    for (i=0; i<max_pixel_val; i= i+1)
    begin
        gray_level[i] <= 0;
        cdf[i] <= 0;
    end    
end

// Data read stage. Image is read serially, in each clock; 1 pixel will be readed.
// This can be enchaned since we have 32 bits wide memory lines to the chips. 
// in this way -> 4 pixels can be read from memory in a clock

always @(posedge clk)
begin
    if ((step_counter == 0) && (read_start == 1))
    begin
        image[x_counter][y_counter] <= pixel_val;
        current_pixel <= pixel_val;
        if (y_counter == width - 1)
        begin
            if(x_counter == height - 1) //continue till all pixels are read. 
            begin
                x_counter <= 0;
                y_counter <= 0;
                step_counter <= step_counter + 1; //increase step counter for next step
            end                
            else
            begin
                x_counter <= x_counter + 1;
                y_counter <= 0;
            end
        end
        else
        begin
            y_counter <= y_counter + 1;
        end            
    end
end


// HISTOGRAM VALUES CALCULATION
// 4 pixels are checked in each clock during the calculation of histogram values of each pixel.
// This can be implemented in a more parallel way. However, more parallel versions require more hardware
// resource for both building the algorithm to generate bitstream and fpga resources. 

always @(posedge clk)
begin
    if (step_counter == 1)
    begin
        gray_level[image[x_counter][y_counter]] = gray_level[image[x_counter][y_counter]] + 1; 
        gray_level[image[x_counter][y_counter + 1]] = gray_level[image[x_counter][y_counter + 1]] + 1; 
        gray_level[image[x_counter][y_counter + 2]] = gray_level[image[x_counter][y_counter + 2]] + 1; 
        gray_level[image[x_counter][y_counter + 3]] = gray_level[image[x_counter][y_counter + 3]] + 1;

        if (y_counter + 4 == width)  //since algorithm will read 4 pixels in each clock, column will shifted by 4 in each clock
        begin
            if (x_counter + 1 == height)
            begin
                y_counter <= 0;
                x_counter <= 0;
                step_counter <= step_counter + 1; //read will be completed after histogram values is calculated for all pixels
            end
            else
            begin
                y_counter <=0;
                x_counter <= x_counter + 1;
            end
        end
        else
        begin
            y_counter <= y_counter + 4;
        end
    end
end

// CDF VALUES CALCULATION
// By starting from the existing non-zero pixel value, 
// CDF for each histogram value is calculated. 

always @(posedge clk)
begin
    if (step_counter == 2)
    begin
        if (y_counter != max_pixel_val -1)
        begin
      	   	y_counter <= y_counter + 1;
            if (gray_level[y_counter] != 0) //without touching the pixels that have 0 pixel value.
            begin
                last_cdf_idx <= y_counter; 
                cdf[y_counter] <= cdf[last_cdf_idx] + gray_level[y_counter];
                current_pixel <= cdf[last_cdf_idx] + gray_level[y_counter];
            end
        end 
        else
        begin
            step_counter <= step_counter + 1;
            y_counter <= 0;
        end
    end 
end

// FINDING NON-ZERO CDF MIN PİXEL VALUE
always @(posedge clk)
begin
    if (step_counter == 3)
    begin
        if ((cdf[y_counter] == 0) && (min_found == 0))
        begin
            y_counter <= y_counter + 1; // increment it itll reaching the non zero CDF value. 
        end
        else
        begin
            cdf_min <= cdf[y_counter]; //after finding cdf_min for the lowest non-zero pixel
            min_found <= 1;         //assign the result to the cdf-min
            step_counter <= step_counter + 1;
            y_counter <= 0;
            x_counter <= 0;
        end
    end
end

// HISTOGRAM EQUALIZATION PART
// Histogram equalization part will operates with the formula implemented below. 
// With respect to this formula, each pixel's new equalized pixel value is calculated and remapped to the corresponding pixel
// This step is again implemented in 4-way such that 4 pixels will be evaluated and remapped in each cycle. 
always @(posedge clk)
begin
    if (step_counter == 4)
    begin
        image[x_counter][y_counter] = ((cdf[image[x_counter][y_counter]] - cdf_min)*(max_pixel_val -1))/(width*height - cdf_min); 
        image[x_counter][y_counter + 1] = ((cdf[image[x_counter][y_counter + 1]] - cdf_min)*(max_pixel_val -1))/(width*height - cdf_min); 
        image[x_counter][y_counter + 2] = ((cdf[image[x_counter][y_counter + 2]] - cdf_min)*(max_pixel_val -1))/(width*height - cdf_min); 
        image[x_counter][y_counter + 3] = ((cdf[image[x_counter][y_counter + 3]] - cdf_min)*(max_pixel_val -1))/(width*height - cdf_min);
        if (y_counter + 4 == width)
        begin
            if (x_counter == height - 1)
            begin
                step_counter <= step_counter + 1;
                y_counter <= 0;
                x_counter <= 0;
            end
            else
            begin
                x_counter <= x_counter + 1;
                y_counter <= 0;
            end            
        end
        else
        begin
            y_counter <= y_counter + 4;
        end
    end
end

// Storing the calculated histogram values.
always @(posedge clk)
begin
    if (step_counter == 5)
    begin
        if ((y_counter < width) && (x_counter < height))
        begin
            finish <= 1;
            out_pixel_val <= image[x_counter][y_counter];
            y_counter <= y_counter + 1;
            if (y_counter == width-1)
            begin
                y_counter <= 0;
                x_counter <= x_counter +1;
            end            
        end
        else
        begin
            step_counter <= step_counter + 1;
            y_counter <= 0;
            x_counter <= 0;
        end
    end     
end
endmodule













