# InternshipAtSundance

This repository includes different Neural Network algorithm configurations implemented in http://yann.lecun.com/exdb/mnist/ page and histogram equalization algorithm. 

1) Neural Network algorithm
  - There are three different network configurations as:
    i) 2-layer NN, 300 hidden units, mean square error
    ii) 3-layer NN, 300+100 hidden units with means square error and
    iii) 3-layer NN, 500+300 HU, softmax, cross entropy, weight decay.
  
  The source code of the NN implementation includes the explanation for the architecture configurations, loss functions and activations. Also, resultant biases and weights are stored in each folder in DeepNeuralNetwork. One can train the algortihms as described in the .readme of DeepNeuralNetwork folder. For more detail, please read the document inside the DeepNeuralNetwork folder. 

2) Histogram Equalization algortihm
  
  The algorithm and a testbench file are implemented in verilog HDL and tested on VCS-1 board provided by Sundance. Also, obtained results are presented in the HiPEAC Student Challange VIII event. Implementation details of the algorithm and results are described in the document inside the Histogram_Equalization folder. 
