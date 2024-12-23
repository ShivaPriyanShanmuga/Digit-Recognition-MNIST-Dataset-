import numpy as np
from keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# we see x_train is a 3D array with depth being 60000[the no. of training eg]
def one_hot(n):
    z = np.zeros(10)
    z[n] = 1
    z = np.array([[i] for i in z])
    return z

x_train_mod = np.array([[matrix.flatten() / 255] for matrix in x_train])
y_train_mod = np.array([one_hot(i) for i in y_train])
x_test_mod = np.array([[matrix.flatten() / 255] for matrix in x_test])
y_test_mod = np.array([one_hot(i) for i in y_test])

# Now x_train and x_test have all been flattened, and they are of the shape (n,784,0), i.e. it is 2 dimensional, depth and

def init_params():
    w1 = np.random.uniform(-0.5, 0.5, size=(16, 784))
    b1 = np.random.uniform(-1, 1, size=(16,1))
    w2 = np.random.uniform(-0.5, 0.5, size=(16, 16))
    b2 = np.random.uniform(-1, 1, size=(16, 1))
    w3 = np.random.uniform(-0.5, 0.5, size=(10, 16))
    b3 = np.random.uniform(-1, 1, size=(10, 1))
    return w1, b1, w2, b2, w3, b3
#def cost(w1, b1, w2, b2, w3, b3)print(w1)

def sigmoid(num):
    return 1 / (1 + (np.e ** (- num)))

def ReLU(num):
    return max(0, num)

def forward_prop(w1, b1, w2, b2, w3, b3, imm, actuall): # the imm is the current image matrix as a column matrix
    z1 = ((w1 @ imm.T) + b1)
    a1 = np.array([sigmoid(i) for i in z1]) # a1 is basically just layer 1 after sigmoid has been applied
    z2 = ((w2 @ a1) + b2)
    a2 = np.array([sigmoid(i) for i in z2]) # a2 is basically just the layer 2 after sigmoid squish
    z3 = ((w3 @ a2) + b3)
    a3 = np.array([sigmoid(i) for i in z3]) # a3 is basically the o/p layer with sigmoid squish
    cost = np.sum((a3 - actuall) ** 2)
    return a1, a2, a3, cost, z1, z2, z3

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def back_prop(w1, b1, w2, b2, w3, b3, a1, a2, a3, actuall, z1, z2, z3, imm, alpha):
    error = a3 - actuall

    dw3 = np.array([sigmoid_derivative(i) for i in z3]) * error @ a2.T
    db3 = np.sum(error * np.array([sigmoid_derivative(i) for i in z3]), axis = 0)

    hidden2_error = w3.T @ (np.array([sigmoid_derivative(i) for i in z3]) * error)
    dw2 = hidden2_error * np.array([sigmoid_derivative(i) for i in z2]) @ a1.T
    db2 = np.sum(hidden2_error * np.array([sigmoid_derivative(i) for i in z2]), axis = 0)

    hidden1_error = w2.T @ (np.array([sigmoid_derivative(i) for i in z2]) * hidden2_error)
    dw1 = hidden1_error * np.array([sigmoid_derivative(i) for i in z1]) @ imm
    db1 = np.sum(hidden1_error * np.array([sigmoid_derivative(i) for i in z1]), axis = 0)
    

    w3 -= dw3 * alpha  # where alpha is the learning rate
    b3 -= db3 * alpha
    w2 -= dw2 * alpha
    b2 -= db2 * alpha
    w1 -= dw1 * alpha
    b1 -= db1 * alpha

    return w1, b1, w2, b2, w3, b3

'''
L1, L2, L3, cost, z1, z2, z3 = forward_prop(w1, b1, w2, b2, w3, b3, x_train_mod[0], y_train_mod[0])
print("cost before : ", cost)

back_prop(w1, b1, w2, b2, w3, b3, L1, L2, L3, y_train_mod[0], z1, z2, z3, x_train_mod[0])

L1, L2, L3, cost, z1, z2, z3 = forward_prop(w1, b1, w2, b2, w3, b3, x_train_mod[0], y_train_mod[0])
print('cost after : ', cost)
'''

#print(forward_prop(w1, b1, w2, b2, w3, b3, x_train_mod[0], y_train_mod[0]))
#print(y_train[0])

#print(x_train_mod[0].T.shape)
# x_train[0].flatten().shape |  this makes it into a flattened list with all the 784 pixels corresponding to themselves~

def gradient_descent(iterations, alpha):
    count = 15
    length = 0
    l = []
    w1, b1, w2, b2, w3, b3 = init_params()
    L1, L2, L3, cost, z1, z2, z3 = forward_prop(w1, b1, w2, b2, w3, b3, x_train_mod[0], y_train_mod[0])
    print("cost before", cost)

    for i in range(iterations):
        L1, L2, L3, cost, z1, z2, z3 = forward_prop(w1, b1, w2, b2, w3, b3, x_train_mod[i], y_train_mod[i])
        if i < count:
            print("cost less 15 : ", cost)
        w1,b1, w2, b2, w3, b3 = back_prop(w1, b1, w2, b2, w3, b3, L1, L2, L3, y_train_mod[i], z1, z2, z3, x_train_mod[i], alpha)
        if i % 1000 == 0:
            print("cost in between : ", cost)
    L1, L2, L3, cost, z1, z2, z3 = forward_prop(w1, b1, w2, b2, w3, b3, x_train_mod[iterations], y_train_mod[iterations])
    print("cost after: ", cost, end='\n\n')


    print('----------')
    print('entering testing phase ...')
    print('----------')

    for k in range(10000):
        L1, L2, L3, cost, z1, z2, z3 = forward_prop(w1, b1, w2, b2, w3, b3, x_test_mod[k], y_test_mod[k])
        l.append(cost)
        length+=1
        print('Avg cost is', sum(l) / length)

gradient_descent(59999, 0.9)

