
import numpy as np
"""
第一步，先计算 A,B,PI,矩阵概率
"""

fileName = 'HMMTrainSet.txt'
A = np.zeros((4, 4))
B = np.zeros((4, 65536))  # 使用ord函数对字符进行编码
PI = np.zeros(4)
status2num={'B':0,'M':1,'E':2,'S':3}



file =  open(fileName,encoding='utf-8')
for line in file.readlines():
    wordStatus = []
    words=line.strip().split() #除去前后空格，然后依照中间空格切分为单词 words shi list列表
    for i,word in enumerate(words):
        """
        矩阵B(发射概率)估计
        """
        if  len(word)==1 :
            status = 'S'
            code = ord(word)
            B[status2num['S']][code] += 1

        else:
            status='B'+(len(word)-2)*'M'+'E'
            for j in range(len(word)):
                code = ord(word[j])
                B[status2num[status[j]]][code] += 1
        if i ==0 :
            """
            初始矩阵PI估计
            """
            #   # status 有可能是 ’s' ,有可能是'BMMME',所以要 status[0]
            PI[status2num[status[0]]]+=1

        """
        矩阵A(转移概率)估计
        """
        wordStatus.extend(status)
        for i in range(1, len(wordStatus)):
            # wordStatus获得状态，使用status2num来映射到正确位置
            A[status2num[wordStatus[i - 1]]][status2num[wordStatus[i]]] += 1
file.close()


"""
取对数概率 矩阵PI
"""
total = sum(PI)
for i in range(len(PI)):
    if PI[i] == 0:
        PI[i] = -3.14e+100
    else:
        # 别忘了去取对数
        PI[i] = np.log(PI[i] / total)

"""
取对数概率 矩阵A
"""
for i in range(len(A)):
    total = sum(A[i])
    for j in range(len(A[i])):
        if A[i][j] == 0:
            A[i][j] = -3.14e+100
        else:
            A[i][j] = np.log(A[i][j] / total)


"""
取对数概率 矩阵B
"""
for i in range(len(B)):
    total = sum(B[i])
    for j in range(len(B[i])):
        if B[i][j] == 0:
            B[i][j] = -3.14e+100
        else:
            B[i][j] = np.log(B[i][j] / total)



"""
维特比算法预测
"""

""" 测试样例 """
test_article=[]
file = open('test.txt',encoding='utf-8')
for line in file.readlines():
    line = line.strip()
    test_article.append(line)

for line in test_article:
    # 定义delta，psi
    # delta一共长度为每一行长度，每一位有4种状态
    print(len(line))
    delta = [[0,0,0,0] for i in range(len(line))]
    psi = [[0,0,0,0] for i in range(len(line))]

    for t in range(len(line)):
        if t == 0:
            # 初始化psi
            psi[t][:] = [0, 0, 0, 0]
            for i in range(4):
                # !!! 注意这里是加号，因为之前log处理了
                delta[t][i] = PI[i] + B[i][ord(line[t])]
        else:
            for i in range(4):
                temp = [delta[t - 1][j] + A[j][i] for j in range(4)]
                delta[t][i] = max(temp) + B[i][ord(line[t])]
                psi[t][i] = temp.index(max(temp))

    status=[] #用于保存最优状态链





