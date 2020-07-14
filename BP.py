

import numpy as np


class BP:

    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def d_loss(self,y,y_pre):
        return y - y_pre

    def sigmod(self,x):
        return 1/(1+np.exp(-x))

    def out_h1(self,a,b):
        tmp = self.w1*a +self.w2*b+self.b1
        return self.sigmod(tmp)

    def out_h2(self,a,b):
        tmp = self.w3*a +self.w4*b+self.b2
        return self.sigmod(tmp)

    def out_o(self,h1,h2):
        tmp = h1*self.w5 + h2*self.w6 +self.b3
        return self.sigmod(tmp)

    """ 循环1000次 """
    def trian(self, a, b, y):

        tmp = []
        learn_rate = 0.04

        for i in range(10000):
            for A,B,Y in zip(a,b,y):

                h1 = self.out_h1(A, B)
                h2 = self.out_h2(A, B)
                o = self.out_o(h1, h2)

                self.w5 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * h1
                self.w6 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * h2
                self.b3 += learn_rate *self.d_loss(Y, o) * o * (1 - o)

                self.w4 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * self.w6 * h2 * (1 - h2) * B
                self.w3 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * self.w6 * h2 * (1 - h2) * A
                self.b2 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * self.w6 * h2 * (1 - h2)

                self.w2 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * self.w5 * h1 * (1 - h1) * B
                self.w1 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * self.w5 * h1 * (1 - h1) * A
                self.b1 += learn_rate *self.d_loss(Y, o) * o * (1 - o) * self.w5 * h1 * (1 - h1)

                tmp.append( abs(o-Y) )
                print(abs(o-Y))


        return tmp


if __name__ == '__main__':
    A = np.array([-2,25,17,-15])
    B = np.array([-1,6,4,-6])
    y = np.array([1,0,0,1])
    a = BP()
    test = a.trian(A,B,y)






