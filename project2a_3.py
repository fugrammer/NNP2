from load import mnist
import numpy as np
import pylab

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# 1 convolution layer, 1 max pooling layer and a softmax layer
#
np.random.seed(10)
batch_size = 128
noIters = 100

def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def model(X, w1, b1, w2, b2, w3, b3, w4, b4):
    pool_dim = (2, 2)

    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    o1 = pool.pool_2d(y1, pool_dim,ignore_border=False)

    y2 = T.nnet.relu(conv2d(o1,w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    o2 = pool.pool_2d(y2,pool_dim,ignore_border=False)

    _o2 = T.flatten(o2, outdim=2)
    pyx2 = T.nnet.relu(T.dot(_o2,w3) + b3)

    pyx = T.nnet.softmax(T.dot(pyx2, w4) + b4)

    return y1, o1, y2, o2, pyx2, pyx

def RMSprop(cost, params, lr=0.001, decay=1e-4, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc=theano.shared(p.get_value()	* 0.)
        acc_new =rho * acc + (1-rho) * g**2
        gradient_scaling = T.sqrt(acc_new+epsilon)
        g = g / gradient_scaling
        updates.append((acc,	acc_new))
        updates.append((p,	p	- lr *	(g+	decay*p)))
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]


X = T.tensor4('X')
Y = T.matrix('Y')

# Convo 1
w1, b1 = init_weights_bias4((15, 1, 9, 9), X.dtype)
# Convo 2
w2, b2 = init_weights_bias4((20, 15, 5, 5), X.dtype)
# F3
w3, b3 = init_weights_bias2((20*3*3, 100), X.dtype)
# F4
w4, b4 = init_weights_bias2((100, 10), X.dtype)

# Model 2-layers CNN
y1, o1, y2, o2, py_x2, py_x  = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]

updates = RMSprop(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1, y2, o2], allow_input_downcast=True)

a = []
for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    teX, teY = shuffle_data (teX, teY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train(trX[start:end], trY[start:end])
    a.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    print("Iter: %d, accuracy: %0.2f" %(i,a[i]))

pylab.figure()
pylab.plot(range(noIters), a)
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2a3.png')

w = w1.get_value()
pylab.figure()
pylab.gray()
for i in range(15):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
# pylab.title('Layer 1 filters learned')
pylab.savefig('figure_2a3_layer1filter.png')

w = w2.get_value()
for i in range(20):
    pylab.figure()
    pylab.gray()
    for o in range(15):
        pylab.subplot(5, 5, o+1); pylab.axis('off'); pylab.imshow(w[i,o,:,:].reshape(5,5))
    # pylab.title('Layer 2 filter '+str(i)+' learned')
    pylab.savefig('figure_2a3_layer2filter'+str(i)+'.png')

ind = np.random.randint(low=0, high=2000)
convolved, pooled, convolved2, pooled2 = test(teX[ind:ind+1,:])

pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
#pylab.title('input image')
pylab.savefig('figure_2a3_input.png')

pylab.figure()
pylab.gray()
for i in range(15):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved[0,i,:].reshape(20,20))
#pylab.title('convolved feature maps')
pylab.savefig('figure_2a3_layer1convolvedfm.png')

pylab.figure()
pylab.gray()
for i in range(15):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled[0,i,:].reshape(10,10))
#pylab.title('pooled feature maps')
pylab.savefig('figure_2a3_layer1pooledfm.png')

pylab.figure()
pylab.gray()
for i in range(20):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved2[0,i,:].reshape(6,6))
#pylab.title('convolved feature maps')
pylab.savefig('figure_2a3_layer2convolvedfm.png')

pylab.figure()
pylab.gray()
for i in range(20):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled2[0,i,:].reshape(3,3))
#pylab.title('pooled feature maps')
pylab.savefig('figure_2a3_layer2pooledfm.png')
