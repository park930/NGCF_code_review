import numpy as np
from sklearn.metrics import roc_auc_score

# 정확도 측정 지표 recall구하기 : rank = 예측한 추천 항목 / groud_truth = 실제 선호 항목
def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

# 상위 k개의 추천항목에 대한 Precision 지표 계산
# Precision = 추천 항목 중 실제 관련 있는 항목 비율 측정
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    # k값이 1 이상인지 확인 (k보다 작으면 계산 의미 없음)
    assert k >= 1
    # 리스트 r을 numpy로 변환해서 효율적 연산 가능하도록 (상위 k개 항목만)
    r = np.asarray(r)[:k]
    # 상위 k개 항목에서 Precision 값 계산해 반환
    return np.mean(r)

# 추천 항목에 대해 Average Precision을 계산
def average_precision(r,cut):
    # AP = Precision을 누적해 평균 구함 (Precision-Recall 곡선의 아래 부분)
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    # r을 Numpy로 변환
    r = np.asarray(r)
    # 상위 cut개에 대해 Precision-k를 계산
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    # 관련성 있는 항목이 없으면 AP값을 0으로
    if not out:
        return 0.
    # AP값 계산 = 고려할 항목의 최대 개수는 cut과 관련 있는 것 중 더 작은 것으로
    return np.sum(out)/float(min(cut, np.sum(r)))

# 추천 결과에 대해 MAP계산
def mean_average_precision(rs):
    # 여러 추천 결과에 대해 AP값을 평균내서
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


# Discounted Cumulative Gain을 계산
def dcg_at_k(r, k, method=1):
    # DCG는 추천 시스템에서 관련 항목이 상위에 있을 수록 높은 점수 주는 지표
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    # 상위 K개만 슬라이싱
    r = np.asfarray(r)[:k]
    # r이 비어 있지 않을 때,
    if r.size:
        if method == 0:
            # 두번째 항목 이후 점수를 log스케일로 감소된 가중치로 나눔
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            # 모든 항목에 대해 가중치 적용 (로그값 계산에 항목 순위가 2부터 시작)
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

# 주어진 추천 항목과 정답을 비교하여 Normalized Discounted Cumulative Gain 계산
def ndcg_at_k(r, k, ground_truth, method=1):
    # DCG값을 정답인 DCG값으로 정규화한 Metric
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    # [최대 k개의 정답 항목에 대해 이상적인 관련성 점수 리스트 생성]
    # 정답 항목이 k보다 많으면 -> 상위 k개 점수를 모두 1로
    if len(GT) > k :
        sent_list = [1.0] * k
    
    # 상위 점수는 1 , 나머지는 0
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    
    # 최대 DCG계산
    dcg_max = dcg_at_k(sent_list, k, method)
    
    # 최대 DCG가 0이면, nDCG로 0으로
    if not dcg_max:
        return 0.
    
    # 실제 추천결과의 DCG / 최대 DCG (정규화 위해)
    return dcg_at_k(r, k, method) / dcg_max


# 상위 k개 항목에서 Recall 계산
def recall_at_k(r, k, all_pos_num):
    # if all_pos_num == 0:
    #     return 0
    r = np.asfarray(r)[:k]
    # all_pos_num = 실제 관련 있는 전체 항목 수 
    return np.sum(r) / all_pos_num

# 상위 k개 중 하나라도 관련 있는 항목 있는지 확인(Hit Ratio)
def hit_at_k(r, k):
    r = np.array(r)[:k]
    # 상위 k개 중에 하나라도 관련 있으면 true로
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

# Precision, Recall로 F1 Score 계산
def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

# 실제 정답, 예측 값으로 AUC 계산
def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res