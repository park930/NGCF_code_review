'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time


# 메인프로그램일 때만
if __name__ == '__main__':

    # gpu_id로 cuda 디바이스를 생성
    args.device = torch.device('cuda:' + str(args.gpu_id))

    # 희소행렬 불러옴 ( 원본, 정규화(자기연결O), 정규화(자기연결X) )
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    # 드롭아웃 적용할 노드 비율 ("0.1"을 실제 실수 0.1로 변환)
    # args는 parser를 통해 여러 하이퍼파라미터를 포함하는 객체
    args.node_dropout = eval(args.node_dropout)
    # 마찬가지로 메세지 드롭아웃 비율 설정
    args.mess_dropout = eval(args.mess_dropout)

    # 모델 생성 (사용자, 아이템 개수, 정규화된 인접행렬, 여러 하이퍼파라미터) -> GPU에 로드
    model = NGCF(data_generator.n_users,
                data_generator.n_items,
                norm_adj,
                args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """

    # 조기종료 = '현재까지 가장 좋은 성능'과 '멈추는 조건' 추적하는 변수 
    cur_best_pre_0, stopping_step = 0, 0
    # 모델의 파라미터 업데이트 (Adam Optimizer)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 여러 성능 지표 기록위해 (매 epoch마다 계산)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    
    # 훈련을 epoch 횟수만큼 반복
    for epoch in range(args.epoch):
        t1 = time()
        # 손실, 행렬 분해 손실, 임베딩 손실
        loss, mf_loss, emb_loss = 0., 0., 0.
        # batch 수 계산 (훈련 Data를 batch크기로 나누려고)
        n_batch = data_generator.n_train // args.batch_size + 1

        # 각 배치에
        for idx in range(n_batch):
            # [사용자,긍정아이템,부정아이템] 샘플링함
            users, pos_items, neg_items = data_generator.sample()
            # 모델을 호출해 각 배치에 대한 사용자 및 아이템의 임베딩을 계산
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                        pos_items,
                                                                        neg_items,
                                                                        drop_flag=args.node_dropout_flag)

            # 'create_bpr_loss'함수 = BPR Loss 계산
            # batch_loss(총 손실), batch_mf_loss(행렬 분해 손실), batch_emb_loss(임베딩 손실)
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                            pos_i_g_embeddings,
                                                                            neg_i_g_embeddings)
            
            # 이전 Gradient값 초기화
            optimizer.zero_grad()
            # 손실에 대한 역전파 수행해 Gradient계산
            batch_loss.backward()
            # 계산한 Gradient로 모델 파라미터 업데이트
            optimizer.step()

            # 각 배치의 손실을 누적해 전체 손실 계산
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        # Epoch마다 성능 출력
        if (epoch + 1) % 10 != 0:
            # verbose = 출력 주기 설정
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        
        t2 = time()
        
        # test_set으로 모델 성능 테스트
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        # 각 epooch마다 성능 지표 기록
        loss_loger.append(loss)             # 총 손실
        rec_loger.append(ret['recall'])     # Recall
        pre_loger.append(ret['precision'])  # Precision
        ndcg_loger.append(ret['ndcg'])      # NDCG
        hit_loger.append(ret['hit_ratio'])  # hit ratio

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                    'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                    (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        # 조기종료 (성능 개선 X = 훈련 종료)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        
        # 성능 향상 없으면 종료 
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.

        # [모델 가중치 저장]
        # recall값이 최상이면, 모델 가중치를 저장
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    # 각 성능 지표를 numpy 배열로 
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    # 최고 성능 지표를 확인 ( )
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                '\t'.join(['%.5f' % r for r in pres[idx]]),
                '\t'.join(['%.5f' % r for r in hit[idx]]),
                '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)