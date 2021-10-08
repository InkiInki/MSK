"""
作者: 因吉
邮箱: inki.yinji@gmail.com
创建日期：2021 0904
近一次修改：2021 0913
"""


import numpy as np
from sklearn.metrics import euclidean_distances, zero_one_loss, log_loss
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from MIL import MIL
from B2B import IsolationForest, isk
from utils import get_k_cv_idx, kernel_rbf


class DS2K(MIL):
    """
    双重空间集合核
    """
    def __init__(self, data_path, po_N=None, ne_N=None, weight=(1, 1),
                 kernel_type="msk", isk_num_node=16, isk_num_tree=10,
                 k=10, save_home="../Data/Distance/", bag_space=None):
        """
        :param data_path:           数据集的存储路径
        :param po_N:                正空间大小
        :param ne_N:                负空间大小
        :param weight:              映射后权重
        :param kernel_type:         距离度量或者核函数的类型
        :param isk_num_node:        isk中每棵树的结点数量
        :param isk_num_tree:        isk中森林中的树的数量
        :param k:                   k折交叉验证
        :param save_home:           距离矩阵或者核矩阵的存储路径
        :param bag_space:           可以传入的包及标签空间空间
        """
        super(DS2K, self).__init__(data_path, save_home, bag_space)
        self._po_N = po_N
        self._ne_N = ne_N
        self._weight = weight
        self._kernel_type = kernel_type
        self._isk_num_node = isk_num_node
        self._isk_num_tree = isk_num_tree
        self._k = k
        self._dis = None
        if self._kernel_type != "msk" and self._kernel_type != "isk":
            self.__pre_kernel()

    def __forest(self, idx):
        """"""

        FOREST = []
        # 获取实例空间
        ins_space, _, _ = self.get_sub_ins_space(idx)
        # 获取实例空间的大小
        space_size = len(ins_space)
        # 设定isk树的结点数量
        num_node = min(self._isk_num_node, space_size)
        for i in range(self._isk_num_tree):
            random_idx = np.random.permutation(space_size)[:num_node]
            FOREST.append(IsolationForest(ins_space[random_idx]).tree_)

        return FOREST

    def __kernel(self, VECTOR, FOREST):
        """"""

        if self._kernel_type != "msk" and self._kernel_type != "isk":
            return self._dis

        KERNEL = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if self._kernel_type == "msk":
                    KERNEL[i][j] = KERNEL[j][i] = self._weight[0] * kernel_rbf(VECTOR[i][0], VECTOR[j][0], gamma=1) +\
                                                  self._weight[1] * kernel_rbf(VECTOR[i][1], VECTOR[j][1], gamma=1)
                elif self._kernel_type == "isk":
                    KERNEL[i][j] = KERNEL[j][i] = isk(self.bag_space[i][0][:, :-1],
                                                      self.bag_space[j][0][:, :-1], FOREST)

        return KERNEL

    def __mapping(self, po_ins, ne_ins):
        """
        获取每个包的映射向量
        """
        VECTOR = []
        po_SHAPE, ne_SHAPE = po_ins.shape[0], ne_ins.shape[0]
        for i in range(self.N):
            bag = self.bag_space[i][0][:, :-1]
            VECTOR.append([euclidean_distances(bag, po_ins).mean(0).reshape(1, po_SHAPE),
                           euclidean_distances(bag, ne_ins).mean(0).reshape(1, ne_SHAPE)])

        return VECTOR

    def __pre_kernel(self):
        """
        对于ave_hausdorff这样的核，先预先计算以减少时间开销，但是多次实验以时间对比时，不能够忽略时间开销
        """
        from B2B import B2B
        mi_matrix = self.get_dis_matrix() if self._kernel_type == "mig" else None
        mean_cov = self.get_mean_cov() if self._kernel_type == "mad" else None
        self._dis = B2B(self.data_name, self.bag_space, b2b_type=self._kernel_type, mi_matrix=mi_matrix,
                        mean_cov=mean_cov,
                        b2b_save_home="D:/Data/TempData/DisOrSimilarity/").get_dis()

    def main(self):
        """"""
        tr_idxes, te_idxes = get_k_cv_idx(self.N, self._k)
        # 获取正负包标签
        po_lab, ne_lab = np.max(self.bag_lab), np.min(self.bag_lab)
        # 初始化单实例分类器
        classifier = SVC(kernel="precomputed")
        # 预测标签
        HAT_Y, Y = [], []
        for i, (tr_idx, te_idx) in enumerate(zip(tr_idxes, te_idxes)):
            # print("{}-th fold.".format(i))
            # 获取包标签
            tr_lab = self.bag_lab[tr_idx]
            # 获取正负包索引
            po_idx, ne_idx = np.where(tr_lab == po_lab)[0], np.where(tr_lab == ne_lab)[0]
            """寻找最负代表性实例"""
            # 获取负包中所有的实例
            ne_ins, _, _ = self.get_sub_ins_space(ne_idx)
            # 计算选取的代表性实例的个数
            po_N = 10 if self._po_N is None else self._po_N
            ne_N = 10 if self._ne_N is None else self._ne_N
            k_means = MiniBatchKMeans(n_clusters=ne_N)
            # 选取最负代表性实例
            k_means.fit(ne_ins)
            ne_ins = k_means.cluster_centers_
            """寻找最正代表性实例"""
            po_ins = []
            # 找到每个包中最负代表性实例
            for j in po_idx:
                # 获取当前包
                bag = self.bag_space[j][0][:, :-1]
                # 获取当前包中与最负代表性实例距离最远的实例的索引
                idx = euclidean_distances(bag, ne_ins).sum(1).argmax()
                # 记录在最正实例中
                po_ins.append(bag[idx].tolist())
            po_ins = np.array(po_ins)
            # 找到指定数量的最正实例
            dis = euclidean_distances(po_ins).sum(0)
            max_dis_idx = np.argsort(dis)[::-1]
            po_ins = po_ins[max_dis_idx[:po_N]]
            # 获取双重空间表示
            VECTOR = 0 if self._kernel_type != "msk" else self.__mapping(po_ins, ne_ins)
            # 获取孤立森林用于构建ISK矩阵
            FOREST = 0 if self._kernel_type != "isk" else self.__forest(tr_idx)
            """计算核矩阵"""
            KERNEL = self.__kernel(VECTOR, FOREST)[:, tr_idx]
            # 获得训练核矩阵和测试核矩阵
            tr_KERNEL, te_KERNEL = KERNEL[tr_idx], KERNEL[te_idx]
            """训练"""
            # 训练模型
            classifier.fit(tr_KERNEL, self.bag_lab[tr_idx])
            # 标签预测
            hat_te_y = classifier.predict(te_KERNEL)
            HAT_Y.extend(hat_te_y.tolist())
            Y.extend(self.bag_lab[te_idx].tolist())
        # 返回分类结果
        return zero_one_loss(Y, HAT_Y), log_loss(Y, HAT_Y)


def test():
    file_name = "D:/Data/OneDrive/文档/Code/MIL1/Data/Web/web9+.mat"
    # po_label = 7
    # data_type = "mnist"
    # file_name = data_type + str(po_label) + ".none"
    # data_path = "D:/Data/OneDrive/文档/Code/MIL1/Data"
    # data_path = "D:/Data/"
    # from BagLoader import BagLoader
    # bag_space = BagLoader(seed=1, po_label=po_label, data_type=data_type, data_path=data_path).bag_space
    print(file_name.split("/")[-1])
    svm = DS2K(file_name, kernel_type="min", weight=(1, 1), bag_space=None)
    acc, f1 = [], []
    for i in range(5):
        per = svm.main()
        acc.append(per[0])
        f1.append(per[1])
        print(per)
    print(np.sum(acc) / 5, np.std(acc, ddof=1))
    print(np.sum(f1) / 5, np.std(f1, ddof=1))


if __name__ == '__main__':
    test()
