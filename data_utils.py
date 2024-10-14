import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import torch


def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid

    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
                                (train_list[:, 0], train_list[:, 1])), dtype='float64', \
                               shape=(n_user, n_item))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                                  (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                                shape=(n_user, n_item))  # test_groundtruth

    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return len(self.data)


class DiffusionData_sec_hop(Dataset):
    def __init__(self, data):
        # 计算每个用户的二跳信息
        hop2 = self.get_2hop_item_based(torch.tensor(data, dtype=torch.float32))

        # 将 hop2 信息转换为 FloatTensor
        self.data = torch.FloatTensor(hop2)

    def get_2hop_item_based(self, data):
        # 初始化空张量
        sec_hop_infos = torch.empty(len(data), len(data[0]))  # [n_user, n_item]

        # 对所有用户的物品交互信息按列求和，得到一个物品的交互总数向量，然后除以用户数 n_user
        sec_hop_inters = torch.sum(data, dim=0) / len(data)

        for i, row in enumerate(data):
            # 找到当前用户未交互的物品索引（交互数接近0）
            zero_indices = torch.nonzero(row < 0.000001).squeeze()
            if i % 1000 == 0:
                print(f"Processing user {i}")

            # 将二跳交互信息赋给当前用户
            sec_hop_infos[i] = sec_hop_inters
            # 将用户未交互过的物品信息置为 0
            sec_hop_infos[i][zero_indices] = 0

        return sec_hop_infos

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def __len__(self):
        return len(self.data)


class DataDiffusion2(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        return item1, item2

    def __len__(self):
        return len(self.data1)


class DataDiffusion3(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.data3[index]
        return item1, item2, item3

    def __len__(self):
        return len(self.data1)


def get_top_k_similar_pearson(data, k):
    # Step 1: 数据中心化 (每行减去行的平均值)
    # 对数据进行均值中心化处理，计算每行的均值并从数据中减去
    # 这一步是在进行皮尔逊相关系数计算前常见的预处理操作，确保数据中心围绕 0。
    mean_centered_data = data - data.mean(dim=1, keepdim=True)

    # Step 2: 计算协方差矩阵
    # 协方差矩阵是通过将中心化后的数据矩阵与其转置相乘得到的
    # 这是皮尔逊相关系数的一部分，表示各行之间的协方差
    covariance_matrix = torch.mm(mean_centered_data, mean_centered_data.t())

    # Step 3: 计算标准差
    # 计算每行数据的标准差，使用 L2 范数，即 2 范数
    # 标准差用于将协方差矩阵标准化为皮尔逊相关系数
    std_dev = mean_centered_data.norm(p=2, dim=1, keepdim=True)

    # Step 4: 避免除以 0 的情况
    # 为了避免出现某些行的标准差为零（即所有元素都相同或为零），
    # 将标准差为零的行设置为 1，避免在后续计算中出现除零错误
    std_dev[std_dev == 0] = 1

    # Step 5: 计算皮尔逊相关系数矩阵
    # 皮尔逊相关系数矩阵是将协方差矩阵除以每行标准差的外积
    # 这个矩阵中的每个元素表示两行之间的相关性
    pearson_correlation_matrix = covariance_matrix / torch.mm(std_dev, std_dev.t())

    # Step 6: 去除自相关
    # 由于每个用户或物品与自身的相关性为 1，我们需要将对角线元素设置为一个很小的值
    # 防止在后续操作中出现自相关的值被错误地选为最相似的对象
    # 对角线值减去 2，确保这些值无法作为最大值被选出，因为皮尔逊系数的范围是 [-1, 1]
    eye = torch.eye(pearson_correlation_matrix.size(0), device=pearson_correlation_matrix.device)
    pearson_correlation_matrix -= eye * 2  # Subtract 2 which is definitely out of bound for correlation

    # Step 7: 获取每行的 top-k 最大值的索引
    # 使用 torch.topk 函数沿每行（dim=1）获取 top-k 的最大值索引
    # 这些索引表示每个用户或物品与其余 k 个用户或物品最相似
    _, indices = torch.topk(pearson_correlation_matrix, k=k, dim=1)

    return indices
