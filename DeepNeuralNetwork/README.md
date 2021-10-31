# Neural Network Algorithm

This algorithm includes the implementation of the neural network algorithm shared in http://yann.lecun.com/exdb/mnist/. 
There are 3 different architectural configurations for each of the algorithm. 
1) 2-layer NN, 300 hidden units, with mean square error	
2) 3-layer NN, 300+100 hidden units	with mean square error and
3) 3-layer NN, 500+300 HU, softmax, cross entropy, weight decay.

Each of these configurations are trained with the MNIST dataset shared in the Data folder, and biases and weights are stored for each configurations as implied from the names.

Those biases and weights was created to observe power consumption and performance of the each configuration into VCS-1 board. 
However, I switched my focus to Histogram Equalization algorithm because of the HiPEAC Student Challenge. Hence, the obtained biases and weights can be used to observe
performance and power profiling into VCS-1 with the help of LynSyn Lite. 


To run each of those algortihms separately: 
  - for the first configuration ..$ python3 main.py --2_layered_architecture 
  - for the second configuration ..$ python3 main.py --3_layered_architecture 
  - for the third configuration ..$ python3 main.py --3_layered_architecture_with_CE 

  Also, each of those can be tested by just typing ..$ python3 main.py
  
 Results for the mentioned architectures are shared below. 
 
 ![Screenshot from 2021-10-31 18-25-01](https://user-images.githubusercontent.com/73446582/139590653-a5689a8e-c438-4b62-a80c-e18008cdff7e.png)
