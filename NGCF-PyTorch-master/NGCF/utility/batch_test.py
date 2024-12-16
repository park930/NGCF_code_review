'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq

# 시스템의 cpu코어 개수를 가져와 이를 절반으로 나눔
cores = multiprocessing.cpu_count() // 2

# 외부에서 전달되는 명령어 parsing해서 agrs에서 저장
args = parse_args()

# 명령어 인자로 전달된 Ks를 문자열 -> 리스트 변환.
# Ks는 추천 아이템 상위 K 리스트를 나타냄
Ks = eval(args.Ks)

# data 객체 초기화
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
# n_users = 총 유저 수, n_items = 총 아이템 수를 변수에 저장
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
# n_trains = 학습 / n_test = 테스트 가져옴
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
# 배치 크기를 args.batch_size에 가져옴 (학습/평가에 사용할 데이터의 크기 나타냄)
BATCH_SIZE = args.batch_size

# 유저별 추천 리스트 생성, 평가함
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    # 추천 점수를 저장할 dict를 초기화
    item_score = {}
    
    # 테스트에 포함된 모든 아이템 점수를 item_score에 저장 
    for i in test_items:
        # item_score = 평가할 아이템 리스트 / rating = 각 아이템 추천 점수
        item_score[i] = rating[i]

    # Ks 중에 max 가져옴
    K_max = max(Ks)

    # heapq.nlargest = 아이템 중에 상위 k_max개 반환
    # dict인 item_score에서 점수 기준으로 상위 k_max 아이템 id 추출
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    # 상위 추천 결과 저장할 리스트 
    r = []
    # 상위 추천 리스트의 각 요소에 대해
    for i in K_max_item_score:
        # 유저 긍정 test 아이템에 해당 요소가 있는지 확인
        if i in user_pos_test:
            # 포함 = 1
            r.append(1)
        else:
            r.append(0)
    
    auc = 0.
    return r, auc


# 아이템의 추천 점수와 유저의 긍정 test 데이터를 기반으로 AUC 계산
def get_auc(item_score, user_pos_test):
    # dict인 item_score에서 value 기준으로 정렬
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    # 내림차순으로
    item_score.reverse()
    # 정렬된 item_score에서 아이템 ID만 추출해서 리스트로
    item_sort = [x[0] for x in item_score]
    # 정렬된 item_score에서 아이템 점수만 추출해서 리스트로
    posterior = [x[1] for x in item_score]

    # 긍정 test 아이템 여부 기록할 리스트
    r = []

    # 정렬된 아이템 리스트 순환
    for i in item_sort:
        # i가 유저 긍정 test 아이템에 있는지 확인
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    
    # r = 실제, prediction = 예측 -> AUC 계산
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

# 정렬로 추천 리스트 생성 후, 상위 K개의 결과를 평가
def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    # 추천 점수 저장할 dict
    item_score = {}

    # test의 모든 아이템 점수를 item_score에 저장
    for i in test_items:
        # rating = 해당 아이템의 추천 점수
        item_score[i] = rating[i]

    K_max = max(Ks)

    # 상위 K_max개의 점수를 가진 아이템 반환
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    
    # 상위 추천 결과 기록할 리스트
    r = []

    # 상위 추천 리스트 각 요소가 user 긍정 테스트 data에 포함되었는지 확인 
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    
    # AUC 계산
    auc = get_auc(item_score, user_pos_test)
    return r, auc

# 추천 결과 r, 유저 긍정 테스트 아이템을 통해 성능 계산
def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    
    # Ks의 각 요소에 대해
    for K in Ks:
        # 각 성능 지표 계산
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


# 특정 유저의 추천 결과 생성 + 성능 계산
# x = 유저, 해당 유저의 추천 점수 있는 튜플 
def test_one_user(x):
    # user u's ratings for user u
    # rating = 유저와 모든 아이템에 대한 추천 점수
    rating = x[0]

    #uid
    u = x[1]

    #user u's items in the training set
    try:
        # 유저의 학습 data에서 쓴 Item List
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    #user u's items in the test set
    # 유저의 테스트 data에서 긍정적으로 평가한 Item List
    user_pos_test = data_generator.test_set[u]

    # 모든 아이템 ID있는 집합
    all_items = set(range(ITEM_NUM))

    # 테스트 Item 리스트 생성 (학습 데이터에 포함 안된것만)
    test_items = list(all_items - set(training_items))
    
    # 테스트 방식에 맞춰서 추천 생성 방식 설정
    if args.test_flag == 'part':
        # part인 경우, ????????????????????????????????
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    # 테스트 아이템의 추천 결과에 대한 성능 평가를 계산하고 반환 
    return get_performance(user_pos_test, r, auc, Ks)


# 전체 유저 추천 결과 계산 + 평가 지표 반환
def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    # result = 테스트 결과 저장
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    # 병렬로 계산하기 위한 프로세스 pool
    pool = multiprocessing.Pool(cores)

    # 한번에 처리할 유저 수
    u_batch_size = BATCH_SIZE * 2
    # 한번에 처리할 아이템 수
    i_batch_size = BATCH_SIZE

    # 테스트에 쓸 User
    test_users = users_to_test
    n_test_users = len(test_users)

    # 필요 배치 수 계산 (전체 유저 / 배치크기)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    # 유저 배치 순차 처리
    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        # 현재 배치의 유저ID들
        user_batch = test_users[start: end]


        # 배치 기반 Test
        if batch_test_flag:
            # batch-item의 수
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            # 유저 배치의 모든 아이템의 추천 점수 저장할 배열
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                # 현재 배치의 아이템 ID 리스트생성
                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    # 유저 임베딩, 아이템 임베딩 생성 (drop_flag마다 다르게)
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                item_batch,
                                                                [],
                                                                drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                item_batch,
                                                                [],
                                                                drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                # 아이템 배치의 추천 점수 저장 
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            # 추천 점수 정확한지 확인
            assert i_count == ITEM_NUM


        # 전체 Item 테스트
        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                            item_batch,
                                                            [],
                                                            drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                            item_batch,
                                                            [],
                                                            drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        # 유저 배치의 추천 점수와 유저 ID 묶음
        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        # 유저별 테스트 성능을 병렬로 계산
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        # 배치별 성능 결과 - 평균 계산
        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

    # 처리된 유저 수가 전체 테스트 유저와 같은지 확인
    assert count == n_test_users
    # 멀티프로세싱 풀 닫음
    pool.close()
    return result
