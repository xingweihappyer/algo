




import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error


""" ui 矩阵 """
def u_i(df):
    """ u i 的数量"""
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    """ 建立 ui 矩阵 """
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1] - 1, row[2] - 1] = row[3]
    return ratings




""" 分割 train, test 随机10个非空 """
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        tmp = ratings[user, :].nonzero()[0]
        """ test_ratings 是 index """
        test_ratings = np.random.choice(tmp,size=10,replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test



""" 相似度计算 使用sk库可以替代 速度也很快 """
def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)



""" 为什么要加权处理 ？？？ 预测评分 来源公式 见paper """
def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


""" TOP N 就算物品偏好 """
def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]  # 选取最相关的40个用户的下标
            for j in range(ratings.shape[1]):
                """ u=1 最相似的40个用户    """
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
    return pred


""" 优化，消除评分偏见 效果不明显，几乎无变化 """
def predict_nobias(ratings, similarity, kind='user'):
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]
    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]

    return pred


""" userCF 推荐列表 """
def usercf_reco_list(ratings, user_pred, k=100):
    n = ratings.shape[0]
    reoc_dict = {}
    for u in range(n):
        not_non = np.nonzero(ratings[u])[0]
        need = np.argsort(user_pred[u])       # 从小到大  切片需要注意
        tmp = [i for i in need if i not in not_non]
        rst = tmp[:-k-1:-1]    # 倒序 选择 k 个
        reoc_dict[u] = rst
    return reoc_dict







if __name__ == '__main__':

    u_data_path = r'/home/xingwei/funbox/funbox_git/funbox-recommendation-system/algo'
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(u_data_path + r'/u.data', sep='\t', names=header)

    ratings =  u_i(df)

    train, test = train_test_split(ratings)

    user_similarity = fast_similarity(train, kind='user')
    item_similarity = fast_similarity(train, kind='item')

    user_s = 1 - pairwise_distances(train, metric='cosine')
    items_s = 1 - pairwise_distances(train.T, metric='cosine')

    item_prediction = predict_fast_simple(train, item_similarity, kind='item')
    user_prediction = predict_fast_simple(train, user_similarity, kind='user')
    print('User-based CF MSE: ' + str(get_mse(user_prediction, test)))
    print('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))

    # top n 的相似中预测评分
    user_pred = predict_topk(train, user_similarity, kind='user', k=40)
    item_pred = predict_topk(train, item_similarity, kind='item', k=40)
    print('Top-k User-based CF MSE: ' + str(get_mse(user_pred, test)))
    print('Top-k Item-based CF MSE: ' + str(get_mse(item_pred, test)))

    user_pred = predict_nobias(train, user_similarity, kind='user')
    item_pred = predict_nobias(train, item_similarity, kind='item')
    print('Bias-subtracted User-based CF MSE: ' + str(get_mse(user_pred, test)))
    print('Bias-subtracted Item-based CF MSE: ' + str(get_mse(item_pred, test)))



    """ userCF 推荐列表 """
    reoc_dict = usercf_reco_list(ratings, user_pred, k=100)


