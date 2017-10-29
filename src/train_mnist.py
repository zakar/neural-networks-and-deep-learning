# code for exercises

import mnist_loader
import time
import random
import numpy as np

def sigmod(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmod_d(x):
    return sigmod(x) * (1 - sigmod(x))

class MnistModel:

    def cost_d(self, a, y):
        return a - y
    
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.bias = [ np.random.randn(y, 1) for y in layer_sizes[1:] ]
        self.weights = [ np.random.randn(x, y) for x, y in zip(layer_sizes[1:], layer_sizes[:-1]) ]

    def train(self, train_data, test_data, epoches, mini_batch, lr):
        for i in xrange(0, epoches):
            random.shuffle(train_data)
            start_time = time.time()
            for j in xrange(0, len(train_data), mini_batch):
                # self.step(train_data[j:j+mini_batch], lr)
                self.step_batch(train_data[j:j+mini_batch], lr)
            print 'train epoch cost %.0fs' % (time.time() - start_time)
            self.eval_test(test_data, i)

    def forward(self, x):
        a = x;
        for b, w in zip(self.bias, self.weights):
            a = sigmod(np.dot(w, a) + b)
        return a

    def eval_test(self, data, epoch):
        matches = 0
        for x, y in data:
            a = self.forward(x)
            matches += np.argmax(a) == y 
        print 'epoch %d eval accuracy %d / %d' % (epoch, matches, len(data))

    def eval_step(self, data, epoch, step):
        matches = 0
        for x, y in data:
            a = self.forward(x)
            matches += np.argmax(a) == np.argmax(y) 
        print 'epoch %d step %d train accuracy %d / %d' % (epoch, step, matches, len(data))


    def step(self, data, lr):
        delta_b = [ np.zeros(b.shape) for b in self.bias ]
        delta_w = [ np.zeros(w.shape) for w in self.weights ]
        for x, y in data:
            bp_delta_b, bp_delta_w = self.bp(x, y)
            delta_b = [b + bdb for b, bdb in zip(delta_b, bp_delta_b) ]
            delta_w = [w + bdw for w, bdw in zip(delta_w, bp_delta_w) ]

        self.bias = [ b-lr*db/len(data) for b, db in zip(self.bias, delta_b) ]
        self.weights = [ w-lr*dw/len(data) for w, dw in zip(self.weights, delta_w) ]

    def step_batch(self, data, lr):
        xs = np.concatenate([ x for x, _ in data ], 1)
        ys = np.concatenate([ y for _, y in data ], 1)
        delta_b, delta_w = self.bp(xs, ys)
        self.bias = [ b-lr*db/len(data) for b, db in zip(self.bias, delta_b) ]
        self.weights = [ w-lr*dw/len(data) for w, dw in zip(self.weights, delta_w) ]

    # mini-batch backprop
    def bp(self, x, y):
        a_s = [x]
        z_s = []
        for b, w in zip(self.bias, self.weights):
            z_s.append(np.dot(w, a_s[-1]) + b)
            a_s.append(sigmod(z_s[-1]))

        delta_b = []
        delta_w = []
        delta = self.cost_d(a_s[-1], y) * sigmod_d(z_s[-1])
        delta_b.insert(0, np.sum(delta, 1, keepdims=True))
        delta_w.insert(0, np.zeros((delta.shape[0], a_s[-2].transpose().shape[1])))
        for l in xrange(0, delta.shape[1]):
            delta_w[0] += np.dot(delta[:,l:l+1], a_s[-2].transpose()[l:l+1,:])
        for i in xrange(len(self.bias)-2, -1, -1):
            delta = np.dot(self.weights[i+1].transpose(), delta) * sigmod_d(z_s[i])
            delta_b.insert(0, np.sum(delta, 1, keepdims=True))
            delta_w.insert(0, np.zeros((delta.shape[0], a_s[i].transpose().shape[1])))
            for l in xrange(0, delta.shape[1]):
                delta_w[0] += np.dot(delta[:,l:l+1], a_s[i].transpose()[l:l+1,:])

        return delta_b, delta_w

if __name__ == '__main__':
    train_data, validate_data, test_data = mnist_loader.load_data_wrapper()
    model = MnistModel([784, 100, 100, 10])
    model.train(train_data, test_data, 1000, 10, 3.0)

