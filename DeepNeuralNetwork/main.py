# Name, Surname: Burak Topçu

import torch
import torchvision.datasets as data # just used to load dataset
import numpy as np 
import progressbar # used to trace execution time 
from sklearn.utils import shuffle # used for shuffling training set for each epoch
import getopt, sys # for taking argument from the command line 

#the sigma activation function is the tanh.  
def sigma(x):
	exp = torch.exp(x)
	neg_exp = torch.exp(-1*x)
	return (exp-neg_exp)/(exp+neg_exp)

#first derivative of the tanh function (sigma function)
def dsigma(x):
    tanh = sigma(x)
    return 1 - torch.pow(tanh, 2)

#softmax activagtion function
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x))

#first derivative of the softmax activation function
def dsoftmax(x):
    return softmax(x)*(1-softmax(x))

#rectified linear unit activation function
def relu(x):
	return torch.maximum(x,torch.zeros(x.shape))

#first derivative of relu
def drelu(x):
	return x>0

#mean square error
def loss(v, t):
    loss = torch.sum(torch.pow(v-t, 2))
    return loss

#first derivative of mse
def dloss(v, t):
    return 2*(v-t)

# This is the backprop for cascading softmax and cross entropy losses. 
# instead of first calculating backprop for cross entropy loss, then 
# pass this result from the derivative of the softmax function,
# we can use below expression where v is the output layer of softmax and
# t is the actual target.

# I write their corresponding mathematical equation and just simplfy it. 
def dceloss(v,t):
	res = v-t
	return res

# forward pass for 2 layered architecture
# dimension for each step is described next to for each line
def forward_pass(w1, b1, w2, b2, x):  
    s1 = torch.mm(w1, x) + b1  # sum = weight1 (300, 784) * each_sample(784, 1) + bias1(300,1)
    x1 = sigma(s1) # after passing from the activation function
    s2 = torch.mm(w2, x1) + b2 # sum = weight2 (10, 300) * second_layers_input(300, 1) + bias1(10,1)
    x2 = sigma(s2) # after passing from the activation function
    return x, s1, x1, s2, x2

# backward pass for 2 layered architecture
# dimensions for each step are described next to for each line
def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dw1, db1, dw2, db2):
    dx2 = dloss(x2, t) # first derivative of mse where x2 is the prediction, t is the actual output
    ds2 = torch.mul(dx2, dsigma(s2)) # element wise multiplication after the backward activation of second layer
    									   # using the dsigma
    db2 += ds2		     				   # bias is equal to sums (in each step it is added since updates for bias and weights)
    									   # are done after iteration amount of batch size is completed
    dw2 += torch.mm(ds2, torch.t(x1))      # ds2= (10,1), x1=(300,10) -> dw2 = (10,300)
    dx1 = torch.mm(torch.t(w2), ds2)  	   # dw2 = (10, 300), ds2 = (10, 1) -> dx1 = (300,1) 
    ds1 = torch.mul(dx1, dsigma(s1))       # ds1 = (300, 1), s1 = (300, 1) -> element wise multiplication
    db1 += ds1							
    dw1 += torch.mm(ds1, torch.t(x))       # ds1 (300,1), x = (784,1) -> dw1 = (300, 784)
    return dw1, dw2, db1, db2


def compute_error(test_input, test_target, w1, b1, w2, b2):
    numb_of_error = 0 
    for i in range (0,len(test_target)):
        _, _, _, _, pred = forward_pass(w1, b1, w2, b2, torch.reshape(test_input[i], (len(test_input[i]), 1)))
        if torch.argmax(pred) != torch.argmax(test_target[i]):  # argmax returns with the index of highest value in the specified array, 
                                                                # since we are using one hot encoding, if the element that has value of 
                                                                # maximum in prediction array matches with the index of test_target array, 
                                                                # prediction is true or vice versa
         	numb_of_error += 1
    return numb_of_error/100 # instead of dividing by 10000 and multiplying with 100, I just simply divide by 100

def two_layered_NN(train_input, train_target, test_input, test_target):

    epsilon = 1*(10**(-4)) #for weight initialization

    w1 = torch.zeros(300, 784).normal_(0,epsilon).double()
    b1 = torch.zeros(300,1).normal_(0,epsilon).double()
    w2 = torch.zeros(10, 300).normal_(0,epsilon).double()
    b2 = torch.zeros(10,1).normal_(0,epsilon).double()

    eta = 0.001  # learning rate
    counter = 0	 # just a counter to handle with batches

    for i in progressbar.progressbar(range (0,60000)):  # at total, 100 epoch 
        dw1 = torch.zeros(300,784).double()	            # differential weights and biases
        dw2 = torch.zeros(10, 300).double()
        db1 = torch.zeros(300,1).double()
        db2 = torch.zeros(10,1).double()
        for j in range(counter*100, (counter+1)*100):	# batch size = 100
            x = torch.reshape(train_input[j], (len(train_input[j]),1))	     # just for size matches for the arrays
            t = torch.reshape(train_target[j], (len(train_target[j]), 1))    # just for size matches for the arrays
            x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, x)             # forward propagation for 2 layered architecture
            dw1, dw2, db1, db2 = backward_pass(w1, b1, w2, b2, t,  
                                                           x0, s1, x1, s2, x2, 
                                                           dw1, db1, dw2, db2)  # backward propagation for 2 layered architecture
       
        w1 -= eta * dw1		# updates of the weights and biases with learning rate 
        w2 -= eta * dw2
        b1 -= eta * db1
        b2 -= eta * db2

        if counter < 599:   # controls the completion of 1 epoch, batch size (100) * 600 iteration = 60000 iteration = 1 epoch
            counter += 1
        else:
            counter = 0
            train_input, train_target = shuffle(train_input, train_target)  # shuffle training set and training input
        if i%600 == 0 or i == 29999:	
          	a = compute_error(test_input, test_target, w1, b1, w2, b2)
          	print("Test error: ", a, i)
          	if i % 30000 == 0 and a > 5 and i != 0: # update the learning rate after 50th epoch where the error does not converge
          		eta = eta / 2.5
          	if i % 45000 == 0 and a > 4 and i != 0: # update the learning rate after 75th epoch where the error does not converge
          		eta = eta / 2.5

    np.save('2_layered_arch/2_layered_W1_784_300',w1) # save weights and biases as numpy arrays
    np.save('2_layered_arch/2_layered_W2_300_10',w2)
    np.save('2_layered_arch/2_layered_B1_300_1',b1)
    np.save('2_layered_arch/2_layered_B2_10_1',b2)


# forward propagation function for 3 layered architecture
# the only difference from 2 layered architecture is the 3rd layer
# again, tanh function is used for the activation function
# array sizes are described next to each line respectively
def forward_pass_2(w1, b1, w2, b2, w3, b3, x): 
    s1 = torch.mm(w1, x) + b1  # (300, 784)*(784,1)+(300,1)
    x1 = sigma(s1) # (300,1)	
    s2 = torch.mm(w2, x1) + b2 # (100,300)*(300,1)+(100,1) 
    x2 = sigma(s2) # (100,1)
    s3 = torch.mm(w3, x2) + b3 # (10, 100)*(100,1)+(10,1)
    x3 = sigma(s3) # (10,1)
    return x, s1, x1, s2, x2, s3, x3

# backward propagation function for 3 layered architecture
# the only difference from 2 layered architecture is the 3rd layer
# again, tanh function is used for the activation function
# array sizes are described next to each line respectively

def backward_pass_2(w1, b1, w2, b2, w3, b3, t, x, s1, x1, s2, x2, s3, x3, dw1, db1, dw2, db2, dw3, db3):
    dx3 = dloss(x3, t) # first derivative of mse loss (10,1)
    ds3 = torch.mul(dx3, dsigma(s3)) # elementwise multiplication of s3(10,1) and dx3(10,1)
    db3 += ds3 
    dw3 += torch.mm(ds3, torch.t(x2))
    dx2 = torch.mm(torch.t(w3), ds3) # (100, 10)* (10, 1)
    ds2 = torch.mul(dx2, dsigma(s2)) # elementwise multiplication of dx2(100,1) and s2(100,1)
    db2 += ds2
    dw2 += torch.mm(ds2, torch.t(x1)) # (100,1)*(1,300)
    dx1 = torch.mm(torch.t(w2), ds2) # (300,100)* (100,1)
    ds1 = torch.mul(dx1, dsigma(s1)) # elementwise multiplication of dx1(300,1) and s1(300,1)
    db1 += ds1
    dw1 += torch.mm(ds1, torch.t(x)) # (300,1) * (1, 784)
    return dw1, db1, dw2, db2, dw3, db3

# same as compute_error function for 2 layered architecture architecture apart from the 
# 3rd layer parameters which are w3 and b3. 

def compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3):
    numb_of_error = 0
    for i in range(0, 10000):
        _, _, _, _, _, _, pred = forward_pass_2(w1, b1, w2, b2, w3, b3, 
                                                torch.reshape(test_input[i], (len(test_input[i]), 1)))
        if torch.argmax(pred) != torch.argmax(test_target[i]):
            numb_of_error += 1
    return numb_of_error/100

# 3 layered architecture training starter function
def three_layered_NN(train_input, train_target, test_input, test_target):

    epsilon = 1*(10**(-3))
    # creating weight and bias parameters for layers
    w1 = torch.zeros(300, 784).normal_(0,epsilon).double()
    b1 = torch.zeros(300,1).normal_(0,epsilon).double()
    w2 = torch.zeros(100, 300).normal_(0,epsilon).double()
    b2 = torch.zeros(100,1).normal_(0,epsilon).double()
    w3 = torch.zeros(10, 100).normal_(0,epsilon).double()
    b3 = torch.zeros(10,1).normal_(0,epsilon).double()

    eta = 0.001 # learning rate
    counter = 0 # counter to handle batch

    for i in progressbar.progressbar(range (0,240*100)): #100 epoch

        dw1 = torch.zeros(300,784).double()		# initialize the differential parameters as zero for backprop
        db1 = torch.zeros(300,1).double()
        dw2 = torch.zeros(100, 300).double()
        db2 = torch.zeros(100,1).double()
        dw3 = torch.zeros(10, 100).double()
        db3 = torch.zeros(10,1).double()

        for j in range(counter*250, (counter+1)*250):	 # 250 is the batch size
            x = torch.reshape(train_input[j], (len(train_input[j]),1))
            t = torch.reshape(train_target[j], (len(train_target[j]), 1))
            x, s1, x1, s2, x2, s3, x3 = forward_pass_2(w1, b1, w2, b2, w3, b3, x)
            dw1, db1, dw2, db2, dw3, db3 = backward_pass_2(w1, b1, w2, b2, w3, b3, 
                                                           t, 
                                                           x, s1, x1, s2, x2, s3, x3,
                                                           dw1, db1, dw2, db2, dw3, db3)
        w1 -= eta * dw1	#updates of the weights and biases
        b1 -= eta * db1
        w2 -= eta * dw2
        b2 -= eta * db2
        w3 -= eta * dw3
        b3 -= eta * db3
        
        if counter < 239:	
            counter += 1
        else:
            counter = 0
            train_input, train_target = shuffle(train_input, train_target)        
        if i % (240*5) == 0: 	#updates for the learning rate when the error does not converge after 30th and 50th epochs  
            print("Test error: ", compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3), i)
            if compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3) > 6 and i == 240*30 and i != 0:
                eta = eta / 5
            elif compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3) > 4.5 and i % 240*50 and i != 0:
                eta = eta / 5
    
    np.save('3_layered_arch/3_layered_W1_784_300',w1) # saving model parameters
    np.save('3_layered_arch/3_layered_W2_300_100',w2)
    np.save('3_layered_arch/3_layered_W3_100_10',w3)
    np.save('3_layered_arch/3_layered_B1_300_1',b1)
    np.save('3_layered_arch/3_layered_B2_100_1',b2)
    np.save('3_layered_arch/3_layered_B3_10_1',b3)

# same as the forward pass of 3 layered architecture
# apart from the last layers' activaiton function and layer sizes
# the last layers activation function is softmax activation function
# array sizes are described next to each line respectively for each array 
def forward_prop_3(w1, b1, w2, b2, w3, b3, x): 
    s1 = torch.mm(w1, x) + b1       # (500, 784)*(784, 1) + (500, 1) 
    x1 = sigma(s1)
    s2 = torch.mm(w2, x1) + b2      # (300, 500)*(500, 1) + (300, 1)
    x2 = sigma(s2) 
    s3 = torch.mm(w3, x2) + b3      # (10, 300)*(300, 1) + (10, 1)
    x3 = softmax(s3)                # the third some is passed from the softmax activation (third layers)
    return x, s1, x1, s2, x2, s3, x3

# same as the backward pass of 3 layered architecture
# apart from the last layers' activaiton function, layer sizes and cross entropy loss function

def backward_prop_3(w1, b1, w2, b2, w3, b3, t, x, s1, x1, s2, x2, s3, x3, dw1, db1, dw2, db2, dw3, db3):
    ds3 = dceloss(x3, t) #combine of softmax activation and derivative of the cross entropy for backward prop 
    db3 += ds3	# (10,1)
    dw3 += torch.mm(ds3, torch.t(x2)) # (10,1) * (1, 300)
    dx2 = torch.mm(torch.t(w3), ds3) # (300, 10) * (10, 1)
    ds2 = torch.mul(dx2, dsigma(s2)) # element-wise multiplication of (300, 1) and (300 ,1)
    db2 += ds2 # (300, 1)
    dw2 += torch.mm(ds2, torch.t(x1)) # (300,1) * (1, 500)
    dx1 = torch.mm(torch.t(w2), ds2) # (500,300) * (300,1)
    ds1 = torch.mul(dx1, dsigma(s1)) #(500,1)
    db1 += ds1 #(500,1)
    dw1 += torch.mm(ds1, torch.t(x)) # (500,1)* (1, 784)
    return dw1, db1, dw2, db2, dw3, db3

# weight decay penalty for part C
def l2_penalty(w):
    return w.pow(2) / 2

# starting function for part C
def three_layered_CE_NN(train_input, train_target, test_input, test_target):

    epsilon = 1*(10**(-3))
    # initialization of the model parameters
    w1 = torch.zeros(500, 784).normal_(0,epsilon).double()
    b1 = torch.zeros(500,1).normal_(0,epsilon).double()
    w2 = torch.zeros(300, 500).normal_(0,epsilon).double()
    b2 = torch.zeros(300,1).normal_(0,epsilon).double()
    w3 = torch.zeros(10, 300).normal_(0,epsilon).double()
    b3 = torch.zeros(10,1).normal_(0,epsilon).double()

    eta = 0.001 # learning rate
    counter = 0

    wd = 0.001 # weight decay coefficient

    for i in progressbar.progressbar(range (0,600*100)): # batch size =100, at total, 100 epoch

        dw1 = torch.zeros(500,784).double() # initialization of differential model parameters for each epoch
        db1 = torch.zeros(500,1).double()
        dw2 = torch.zeros(300, 500).double()
        db2 = torch.zeros(300,1).double()
        dw3 = torch.zeros(10, 300).double()
        db3 = torch.zeros(10,1).double()

        for j in range(counter*100, (counter+1)*100):
            x = torch.reshape(train_input[j], (len(train_input[j]),1))
            t = torch.reshape(train_target[j], (len(train_target[j]), 1))
            x, s1, x1, s2, x2, s3, x3 = forward_prop_3(w1, b1, w2, b2, w3, b3, x)
            dw1, db1, dw2, db2, dw3, db3 = backward_prop_3(w1, b1, w2, b2, w3, b3, 
                                                           t, 
                                                           x, s1, x1, s2, x2, s3, x3,
                                                           dw1, db1, dw2, db2, dw3, db3)
        w1 -= eta * (dw1 + l2_penalty(w1)) # weight and bias updates with weight decay condition
        b1 -= eta * (db1 + l2_penalty(b1))
        w2 -= eta * (dw2 + l2_penalty(w2))
        b2 -= eta * (db2 + l2_penalty(b2))
        w3 -= eta * (dw3 + l2_penalty(w3))
        b3 -= eta * (db3 + l2_penalty(b3))
        
        if counter < 599:
            counter += 1
        else:
            counter = 0
            train_input, train_target = shuffle(train_input, train_target)        
        if i % 200 == 0:
        	print("Test error: ", compute_error_3(test_input, test_target, w1, b1, w2, b2, w3, b3), i)
        	
        	# learning rate updates for further epochs,
        	# 3th epoch, 30th epoch and 70th epochs. If error does not converge, I simply divide learning rate with 5
        	if compute_error_3(test_input, test_target, w1, b1, w2, b2, w3, b3) > 10 and ((i == 1800) and (i != 0)):
        		eta = eta / 5

        	elif compute_error_3(test_input, test_target, w1, b1, w2, b2, w3, b3) > 5 and ((i == 18000) and (i != 0)):
        		eta = eta / 5

        	elif compute_error_3(test_input, test_target, w1, b1, w2, b2, w3, b3) > 2 and ((i == 42000) and (i != 0)):
        		eta = eta / 5


    np.save('3_layered_CE_W1_784_500',w1)
    np.save('3_layered_CE_W2_500_300',w2)
    np.save('3_layered_CE_W3_300_10',w3)
    np.save('3_layered_CE_B1_500_1',b1)
    np.save('3_layered_CE_B2_300_1',b2)
    np.save('3_layered_CE_B3_10_1',b3)

def part_1(test_input, test_target):
	w1 = torch.from_numpy(np.load('2_layered_arch/2_layered_W1_784_300.npy'))
	w2 = torch.from_numpy(np.load('2_layered_arch/2_layered_W2_300_10.npy'))
	b1 = torch.from_numpy(np.load('2_layered_arch/2_layered_B1_300_1.npy'))
	b2 = torch.from_numpy(np.load('2_layered_arch/2_layered_B2_10_1.npy'))	
	print('The total error for 2 layered architecture configuration: ', compute_error(test_input, test_target, w1, b1, w2, b2))

def part_2(test_input, test_target):
	w1 = torch.from_numpy(np.load('3_layered_arch/3_layered_W1_784_300.npy'))
	w2 = torch.from_numpy(np.load('3_layered_arch/3_layered_W2_300_100.npy'))
	w3 = torch.from_numpy(np.load('3_layered_arch/3_layered_W3_100_10.npy'))
	b1 = torch.from_numpy(np.load('3_layered_arch/3_layered_B1_300_1.npy'))
	b2 = torch.from_numpy(np.load('3_layered_arch/3_layered_B2_100_1.npy'))	
	b3 = torch.from_numpy(np.load('3_layered_arch/3_layered_B3_10_1.npy'))
	print('The total error for 3 layered architecture configuration: ', compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3))

def part_3(test_input, test_target):
	w1 = torch.from_numpy(np.load('3_layered_arch_with_CE/3_layered_CE_W1_784_500.npy'))
	w2 = torch.from_numpy(np.load('3_layered_arch_with_CE/3_layered_CE_W2_500_300.npy'))
	w3 = torch.from_numpy(np.load('3_layered_arch_with_CE/3_layered_CE_W3_300_10.npy'))
	b1 = torch.from_numpy(np.load('3_layered_arch_with_CE/3_layered_CE_B1_500_1.npy'))
	b2 = torch.from_numpy(np.load('3_layered_arch_with_CE/3_layered_CE_B2_300_1.npy'))	
	b3 = torch.from_numpy(np.load('3_layered_arch_with_CE/3_layered_CE_B3_10_1.npy'))
	print('The total error for 3 layered architecture with cross entropy loss: ', compute_error_2(test_input, test_target, w1, b1, w2, b2, w3, b3))


'''
#importing MNIST data set and training set

mnist_train_set = data.MNIST(root='./data', train=True, download=True, transform=None)
mnist_test_set = data.MNIST(root='./data', train=False, download=True, transform=None)

#specifying NN inputs and their targets for both train and test set

train_input = mnist_train_set.train_data.view(-1, 784).double()
train_t = mnist_train_set.train_labels
test_input = mnist_test_set.test_data.view(-1, 784).double()
test_t = mnist_test_set.test_labels

# converting target files as meaningful outputs
# write 1 for each index for digits [0,9], remaining is 0
# one hot encoding for labels

test_target = torch.zeros((len(test_t), 10), dtype=float)
train_target = torch.zeros((len(train_t), 10), dtype=float)

for i in range (0, 60000):
    train_target[i][train_t[i]] = 1
for i in range (0, 10000):
    test_target[i][test_t[i]] = 1    

#normalize both train and test dataset by dividing them 255

train_input =  train_input / 255
test_input = test_input / 255

np.save('train_input.npy', train_input)
np.save('test_input.npy', test_input)
np.save('train_target.npy', train_target)
np.save('test_target.npy', test_target)
'''

train_input = torch.from_numpy(np.load('data/train_input.npy'))
train_target = torch.from_numpy(np.load('data/train_target.npy'))
test_input = torch.from_numpy(np.load('data/test_input.npy'))
test_target = torch.from_numpy(np.load('data/test_target.npy'))

'''
two_layered_NN(train_input, train_target, test_input, test_target) #training for 2 layer MLP configuration
'''

'''
three_layered_NN(train_input, train_target, test_input, test_target)
'''

'''
three_layered_CE_NN(train_input, train_target, test_input, test_target)
'''


argumentList = sys.argv[1:]

for i in argumentList:
    if i == '--2_layered_architecture':
        part_1(test_input, test_target)
    elif i == '--3_layered_architecture':
        part_2(test_input, test_target)
    elif i == '--3_layered_architecture_with_CE':
        part_3(test_input, test_target)

if argumentList == []:
	part_1(test_input, test_target)
	part_2(test_input, test_target)
	part_3(test_input, test_target)