'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        # nn.Module 클래스의 init 메소드 호출 (PyTorch 기본 기능 상속)
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        # 계산이 실행될 하드웨어 지정함
        self.device = args.device
        # 사용자 / 아이템 임베딩 벡터의 크기 지정
        self.emb_size = args.embed_size
        # batch 크기 지정
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout       #메세지 드롭아웃
        self.batch_size = args.batch_size

        # 정규화된 인접행렬 저장
        self.norm_adj = norm_adj

        # 각 layer의 크기를 저장 (eval = 문자열 형태의 리스트를 실제 리스트로 변환)
        self.layers = eval(args.layer_size)

        # 정규화 하이퍼파라미터 설정 (가중치 크기 제한 -> 과적합 방지)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        # embedding_dict = 사용자-아이템에 대한 초기 임베딩 벡터
        # weight_dict = 네트워크 계층간의 가중치 저장
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        # 희소 인접행렬을 PyTorch 텐서로 변환 후, 디바이스로 이동
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init (각 layer의 입력, 출력 노드 개수를 고려해 적절한 분포에서 값 샘플링)
        initializer = nn.init.xavier_uniform_

        # 사용자-아이템 임베딩을 초기화, 저장
        embedding_dict = nn.ParameterDict({
            # 사용자 임베딩을 위한 텐서 생성 (xavier방식으로 초기화) -> 학습 가능한 파라미터로 등록
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,self.emb_size)))
        })

        # 가중치 / 편향을 저장할 dict 초기화
        weight_dict = nn.ParameterDict()
        # 네트워크 계층의 크기 설정
        layers = [self.emb_size] + self.layers
        
        
        for k in range(len(self.layers)):
            # GCN 가중치 / 편향 초기화
            # k번째 계층의 가중치 행렬을 초기화하고 dict에 추가, Key이름 : 'W_gc_k'로 설정
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],layers[k+1])))})
            # k번째 GCN 계층의 bias 초기화
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
            
            # Bilinear 계층에서의 가중치 행렬 초기화 (GCN 가중치 )
            # k번째 계층의 가중치 행렬을 초기화하고 dict에 추가, Key이름 : 'W_bi_k'로 설정
            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],layers[k+1])))})
            # k번째 Bilinear 계층의 bias 초기화
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        # 초기화된 사용자-아이템 임베딩과 네트워크 가중치 dict 반환
        return embedding_dict, weight_dict

    # 희소행렬 -> PyTorch의 희소 텐서로 변환함
    def _convert_sp_mat_to_sp_tensor(self, X):
        # Coordiante List형식으로
        coo = X.tocoo()
        # PyTorch의 정수 텐서로 변환    -> i = 좌표 제공
        i = torch.LongTensor([coo.row, coo.col])
        # data를 희소 텐서로 변환       -> v = 값 정보 제공
        v = torch.from_numpy(coo.data).float()
        # 희소 텐서 생성, 반환 (shape = 희소행렬 전체 크기)
        return torch.sparse.FloatTensor(i, v, coo.shape)

    # 희소텐서에 dropout 적용 
    def sparse_dropout(self, x, rate, noise_shape):
        # 드롭아웃 비율 기준으로 랜덤 텐서 기준값 생성
        random_tensor = 1 - rate
        # 드롭아웃 마스크 생성 위해 랜덤값 추가
        random_tensor += torch.rand(noise_shape).to(x.device)
        # 드롭아웃 마스크 생성
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        
        # 희소 텐서의 좌표 추출
        i = x._indices()

        # 희소 텐서의 값 추출
        v = x._values()

        # 좌표와 값에 드롭아웃 마스크 적용
        i = i[:, dropout_mask]
        v = v[dropout_mask]

        # 드롭 아웃 적용된 희소 텐서 생성
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        # 드롭 아웃 비율에 따른 보정
        return out * (1. / (1 - rate))

    # BPR손실함수 구현 = 추천시스템에서 사용자-긍정아이템 간의 선호도 학습
    # 긍정 아이템을 부정 아이템보다 선호하도록 학습시킴
    def create_bpr_loss(self, users, pos_items, neg_items):
        
        # 사용자-긍정 아이템과의 상호작용 강도를 점수로 계산
        # 사용자-긍정아이템 벡터의 원소별 곱 -> 내적 결과 합산해서 점수 계산
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        
        # 사용자-부정 아이템과의 상호작용 강도를 점수로 계산
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        # 긍정 점수, 부정 점수의 차이에 대한 로그 시그모이드 값 계산
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        # 평균 로그 시그모이드 값을 기반으로 Matrix Factorization loss 계산
        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        # 정규화 항을 계산 -> 과적합 방지
        # 각 벡터의 L2 norm
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        
        # 정규화 항에 가중치(decay)를 곱하여 최종 임베딩 손실 계산 (배치 단위로 정규화) / 모델 파라미터가 너무 커지지 않도록
        emb_loss = self.decay * regularizer / self.batch_size

        # 최종 손실 반환 (BPR 손실+정규화 손실)
        return mf_loss + emb_loss, mf_loss, emb_loss

    # 사용자 임베딩 벡터, 긍정아이템 임베딩 벡터로 예상 평점 계산
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        # 사용자 임베딩과 아이템 임베딩의 행렬 곱으로 예측 점수 생성
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    # NGCF모델의 순전파 정의 (그래프 임베딩 업데이트 + 사용자-아이템 임베딩 추출)
    # 드롭아웃 적용
    def forward(self, users, pos_items, neg_items, drop_flag=True):
        
        # 드롭아웃 적용 -> 희소행렬의 일부 노드 무작위 제거
        # A_hat = 드롭아웃 적용된 희소 인접 행렬
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    # 희소 행렬에서 값이 0이 아닌 원소 개수
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        # 초기 임베딩 설정 (사용자-아이템 임베딩을 결합한 초기 그래프 임베딩)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        # 각 레이어에서 업데이트된 임베딩을 저장하기 위한 리스트 (1번째 요소 = 초기 임베딩)
        all_embeddings = [ego_embeddings]

        # 레이어 순환과 임베딩 업데이트
        for k in range(len(self.layers)):
            # 각 노드의 이웃으로부터 전달 받은 메세지 집계
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            # 이웃에게 받은 메세지에 선형 변환,편향 적용
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                            + self.weight_dict['b_gc_%d' % k]
            
            # bi messages of neighbors.
            # element-wise product
            # 현재 노드와 이웃의 임베딩 간의 요소별 곱
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # 요소별 곱에 선형 변환, 편향 적용
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation (활성화 함수- Leaky렐루 적용)
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            # 메세지 드롭아웃으로 일부 연결 제거
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            # L2정규화로 각 벡터 정규화 (임베딩 벡터 크기 = 1)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            
            # 각 레이어의 계산된 임베딩을 리스트에 추가
            all_embeddings += [norm_embeddings]

        # [최종 임베딩 계산]
        # 모든 레이어에서 계산된 임베딩을 열 방향으로 연결
        all_embeddings = torch.cat(all_embeddings, 1)
        # 사용자 임베딩 추출
        u_g_embeddings = all_embeddings[:self.n_user, :]
        # 아이템 임베딩 추출
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """
        # [사용자 / 아이템 임베딩 선택]
        # 입력 사용자 인덱스로 사용자 임베딩 추출
        u_g_embeddings = u_g_embeddings[users, :]
        # 긍정 아이템 인덱스로 긍정 아이템 임베딩 추출
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        # 부정 아이템 인덱스로 부정 아이템 임베딩 추출
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        # 최종 사용자,긍정,부정 아이템 임베딩 반환
        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
