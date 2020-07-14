

import math
import random
import numpy as np

class LFM(object):
    def __init__(self):
        """
        f:隐含特征
        n:迭代次数
        """
        self.f = 10
        self.n = 100
        self.alpha = 0.2

        self.lambdas = 0.03

    def data(self):
        user_items = {1: {'a': 1, 'b': -1, 'c': -1, 'd': -1, 'e': 1, 'f': 1, 'g': -1},
                      2: {'a': -1, 'b': 1, 'c': -1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
                      3: {'a': 1, 'b': -1, 'c': 0, 'd': -1, 'e': -1, 'f': -1, 'g': 1},
                      4: {'a': 1, 'b': -1, 'c': -1, 'd': 0, 'e': 1, 'f': 1, 'g': 1},
                      5: {'a': -1, 'b': 1, 'c': 1, 'd': 1, 'e': -1, 'f': -1, 'g': 0},
                      6: {'a': 1, 'b': 0, 'c': -1, 'd': -1, 'e': 1, 'f': -1, 'g': -1}}
        users = {1, 2, 3, 4, 5, 6}
        items = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
        return user_items,users,items

    def init_pq(self):
        p = {}
        q = {}
        user_items, users, items =self.data()
        """ 一般都是将这两个矩阵用随机数填充，但随机数的大小还是有讲究的，根据经验，随机数需要和 1/sqrt(F) 成正比 # 来源 项亮博士 p189 """
        for userid in users:
            p[userid] = np.array([random.random()/math.sqrt(self.f) for x in range(0,self.f)])
        for itme in items:
            q[itme] = np.array([random.random()/math.sqrt(self.f) for x in range(0,self.f)])
        return p,q

    def lfm_para(self):
        p,q = self.init_pq()
        user_items, users, items = self.data()

        for step in range(self.n):
            count = 0
            error = 0
            for user,i in user_items.items():
                for item,rui in i.items():
                    pui =p[user].dot(q[item])
                    e = rui - pui
                    count += 1
                    error += np.power(e,2)
                    d_p = self.alpha * (-e * q[item] + self.lambdas * p[user])
                    d_q = self.alpha * (-e * p[user] + self.lambdas * q[item])
                    p[user] += - d_p
                    q[item] += - d_q
            rmse = np.sqrt(error/count)

            print(rmse)

        return p, q

    def predict(self,userid,item,p,q):
        rating = p[userid].dot(q[item])
        return rating


if __name__ == '__main__':
    a = LFM()
    p,q = a.init_pq()
    last_p, last_q = a.lfm_para()
    rating = a.predict(1,'a',p,q)




















