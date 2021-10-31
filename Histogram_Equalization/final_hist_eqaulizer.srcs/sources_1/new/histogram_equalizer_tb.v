`timescale 1ns / 1ps

module histogram_equalizer_tb;
                              parameter width = 4096, 
                              height = 4096,
                              max_pixel_val = 256;
                   

	// Inputs
    reg [7:0] pixel_val;
    reg ready;
    reg clk;
    reg read_start;

    wire [7:0] out_pixel_val;
    wire finish;
    wire [7:0] current_pixel;
    wire [12:0] y_counter;
    wire [12:0] x_counter;
    wire [3:0] step_counter;
    wire [7:0] cdf_min;
    wire min_found;

	reg read, write;
    reg finish_tb;

	reg [7:0] image [0:width*height-1];
    
	histogram_equalizer dut	(
                            .pixel_val(pixel_val),
                            .out_pixel_val(out_pixel_val),
                            .finish(finish),
                            .clk(clk),
                            .ready(ready),
                            .current_pixel(current_pixel),
                            .read_start(read_start),
                            .y_counter(y_counter),
                            .x_counter(x_counter),
                            .step_counter(step_counter),
                            .cdf_min(cdf_min),
                            .min_found(min_found));
integer out_ptr, i, j;
                        
initial
begin
    i<=0;
    j <= 0;
    clk <= 0;
    ready <= 0;
    read <= 0;
    write <= 0;
    finish_tb <= 0;
    $readmemh("/home/burak/Desktop/histogram_equalizer/final_hist_eqaulizer/lyon_hex.txt", image);
    out_ptr <= $fopen("/home/burak/Desktop/histogram_equalizer/final_hist_eqaulizer/lyon_res.txt","w");
end

always
begin 
    #10 clk = ~clk;
end

always @(posedge clk)
begin
    if (ready == 0)
    begin
        ready <= 1;
        read <= 1;
        write <= 0;
    end
    else if ((read == 1) && (j < width*height))
    begin
        pixel_val <= image[j];
        j <= j + 1;
        read_start <= 1;
    end
    else
    begin
        j <= 0;
        read <= 0;
        ready <= 0;
        read_start <= 0;
    end
end

always @ (posedge clk)
begin
    if ((i < width*height) && (finish == 1))
    begin
        $fwrite(out_ptr, "0x%h\n" ,out_pixel_val);	// Here OUT is the signal you want to write in file
        i <= i + 1;
        if (i == width*height-1)
        begin
            finish_tb <= 1;
        end
    end
    else if ((finish_tb == 1))
    begin
        i <= 0;
    	$fclose(out_ptr);
    end 
end
endmodule
