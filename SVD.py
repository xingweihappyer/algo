
import numpy as np
class SVD(object):

    def tmp(self,diag_s):
        s = np.zeros((u.shape[0], v.shape[0]))
        n = len(diag_s)
        for i in range(n):
           s[i,i] = diag_s[i]
        return s

    def ori(self,u,s,v):
        tmp = u.dot(s)
        rst = tmp.dot(v)
        return rst

    def k_svd(self,a,k=2):
        u, diag_s, v = np.linalg.svd(a)
        s =self.tmp(diag_s)
        uu = u[:,:k]
        ss = s[:k,:k]
        vv = v[:k,:]
        ori = self.ori(uu,ss,vv)
        return uu,ss,vv,ori



if __name__ == '__main__':
    a = np.matrix([[1., 0., 0., 0.],
                   [0., 0., 0., 4.],
                   [0., 3., 0., 0.],
                   [0., 0., 0., 0.],
                   [2., 0., 0., 0.]])
    u, diag_s, v = np.linalg.svd(a)
    A = SVD()
    """ U, S, V 是完全奇异值分解   """
    s =  A.tmp(diag_s)
    """ 是否与原始矩阵相等 """
    a02= A.ori(u,s,v)
    print(a02==a )


    """ 截断奇异值分解 """
    k = 2
    uu, ss, vv, ori =A.k_svd(a,k=2)



