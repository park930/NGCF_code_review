'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []


        # 훈련 데이터파일을 읽어, 유저수/아이템수를 업데이터
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        # 테스트 데이터파일을 읽어, 유저수/아이템수를 업데이터
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        # 유저,아이템 전체 수량 +1
        self.n_items += 1
        self.n_users += 1

        # 데이터 로드와 전처리 후, 현재 데이터에 대한 통계 정보
        self.print_statistics()

        # 사용자-아이템에 대한 희소 행렬 생성 (행 = 사용자 수 , 열 = 아이템 수)
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        # 학습,테스트 데이터 저장 위한 dic 초기화
        self.train_items, self.test_set = {}, {}
        
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        # 사용자와 이이템 간의 상호작용을 희소행렬에 저장
                        self.R[uid, i] = 1.

                    # 현재 사용자에 대한 모든 아이템을 저장 (train파일)
                    self.train_items[uid] = train_items
                
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    
                    uid, test_items = items[0], items[1:]
                    
                    # 마찬가지로, 현재 사용자에 대한 모든 아이템을 저장 (test파일)
                    self.test_set[uid] = test_items


    # 사용자의 그래프 인접 행렬(adjacency matrix)을 로드 및 생성
    def get_adj_mat(self):
        # 저장된 인접행렬 파일 있을 시,
        try:
            t1 = time()
            # 인접행렬 로드
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            # 정규화된 인접행렬 로드
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            # 평균 정규화 인접행렬 로드
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        # 없을 시, 새로 생성
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    # 인정행렬, 정규화된 인접행렬 생성
    def create_adj_mat(self):
        t1 = time()

        # 사용자와 아이템을 합친 노드들의 연결을 나타내는 희소행렬 초기화
        # 희소행렬 크기 = (n users + n items) × (n users + n items). 초기값=0으로 
        adj_mat = sp.dok_matrix((self.n_users + self.n_items , self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # 사용자-아이템인 행렬 R을 리스트(LIL) 형식으로 변환
        # 사용자-아이템 간의 상호작용 나타낼 수 있는 부분 행렬
        R = self.R.tolil()
        
        # 유저-아이템 관계를 왼쪽 아래 부분에 저장
        adj_mat[:self.n_users, self.n_users:] = R
        # 아이템-유저 관계를 오른쪽 위 부분에 저장 =====> adj_mat은 대칭 행렬
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        # 행의 합으로 각 행을 정규화하는 함수
        def mean_adj_single(adj):
            # D^-1 * A (행의 합 기준으로 각 행을 정규화)
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # 행과 열의 합으로 행렬을 정규화하는 함수
        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        # 자기 연결이 있는 평균 정규화된 희소행렬(희소행렬에서 대각선 항 추가)
        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        # 자기 연결 없는 평균 정규화 희소행렬
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
    
    # 각 사용자의 학습에 사용될 부정 아이템 Pool 생성 = 잘못된 상호작용도 학습하도록
    def negative_pool(self):
        t1 = time()
        # train 데이터의 Key에 대해서 반복
        for u in self.train_items.keys():
            # 전체 아이템 중, 사용자와 상호작용 않은 아이템 집합
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            # 사용자 u에 대해 100개의 부정 샘플을 랜덤으로 선택
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    # 배치학습을 위해 긍정/부정 샘플을 샘플링하는 작업 (미니배치 학습 위해)
    def sample(self):
        # 배치 < 사용자 = 랜덤하게 batch크기만큼 사용자 선택
        if self.batch_size <= self.n_users:
            # 현재 유저 중에 중복 없이 랜덤 선택
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            # 배치 > 유저 = 사용자 중복될 수 있어도 랜덤하게 batch 크기만큼 사용자 선택
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # 사용자u에 대해 num개의 긍정 샘플을 샘플링
        def sample_pos_items_for_u(u, num):
            # 사용자 u가 상호작용한 아이템 리스트
            pos_items = self.train_items[u]
            # 사용자 u의 상호작용 아이템 수
            n_pos_items = len(pos_items)
            pos_batch = []
            # pos_batch크기가 num일 때까지 긍정 샘플 추가
            while True:
                if len(pos_batch) == num:
                    break
                # 랜덤 인덱스로 해당 위치의 아이템을 샘플링
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        
        # 사용자u에 대해 num개의 부정 샘플을 샘플링
        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            # neg_items크기가 num일 때까지 부정 샘플 추가
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                # neg_id가 이미 상호작용한 아이템 or 이전에 선택한거면 제외하도록
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        # 부정 샘플풀에서 사용자u에 대해 num개의 부정 샘플을 샘플링함
        def sample_neg_items_for_u_from_pools(u, num):
            # 이미 상호작용한 아이템 빼고 부정아이템 목록 만듬
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            # 중복 없이 랜덤하게 선택
            return rd.sample(neg_items, num)

        # 모든 사용자의 긍정/부정 샘플을 샘플링함
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items


    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    # 주어진 경로에서 sparsity.split 파일을 읽기 or 존재하지 않으면 생성.
    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            # 각 줄마다 분할 상태와 사용자id를 분리하여 저장
            for idx, line in enumerate(lines):
                # 짝수 라인은 분할상태 나타냄
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    # 홀수는 사용자id를 포함 - 공백 기준으로 분리
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        # 파일이 없어서 새로 생성
        except Exception:
            # 사용자의 분할 정보 생성
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                # 분할 상태 기록
                f.write(split_state[idx] + '\n')
                # 분할 상태의 사용자id 리스트를 파일에 기록
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    # generate a dictionary to store (key=n_iids, value=a list of uid).
    # 사용자 데이터 기반으로 희소성을 고려하여 여러 그룹으로 나눔
    def create_sparsity_split(self):
        # 테스트셋의 Key로 사용자id 목록 만듦
        all_users_to_test = list(self.test_set.keys())
        # 사용자별 총 상호작용한 아이템 수 저장
        user_n_iid = dict()

        # 사용자별로 상호작용한 아이템의 총 수(n_iids)를 계산 -> 사용자들 그룹화
        for uid in all_users_to_test:
            # 특정 사용자의 훈련/테스트 아이템 목록을 가져옴
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            # 훈련+테스트 아이템 합쳐서 사용자와 상호작용한 아이템 총 수 계산
            n_iids = len(train_iids) + len(test_iids)

            # n_iids가 없으면 새로운 키 추가, 있으면 해당 리스트에 추가
            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)

        # 분할된 사용자 그룹 저장할 리스트
        split_uids = list()

        # 사용자들을 4개의 그룹으로 분할할 준비
        temp = []   #한 그룹에 포함될 사용자를 일시적 저장
        count = 1   #분할 기준 추적
        fold = 4    #4개의 그룹으로 분할
        n_count = (self.n_train + self.n_test)  #총 아이템 수 계산
        n_rates = 0 #분할 상태의 총 아이템 수 추적

        split_state = []

        # 사용자의 아이템 수를 기준으로 정렬해서 하나씩 처리
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            # 각 그룹에 포함된 사용자의 아이템 수를 n_rates에 누적
            n_rates += n_iids * len(user_n_iid[n_iids])
            # 남은 아이템 수 추적
            n_count -= n_iids * len(user_n_iid[n_iids])
            
            # 사용자의 총 상호작용 아이템 수가 4개의 분할 중 하나에 대해 25% 이상이 되면, 
            # 현재 temp에 포함된 사용자들을 하나의 그룹으로 저장합니다.
            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                # 해당 그룹에 대한 정보 추가
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []   
                n_rates = 0
                # 남은 분할 수 추적
                fold -= 1

            # 마지막 사용자 처리 or 남은 아이템 수가 0이면 마지막 그룹을 split_uids에 추가
            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
