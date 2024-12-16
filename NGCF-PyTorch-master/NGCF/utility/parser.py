'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    # 커멘드라인의 인수를 정의하고 처리 (description = 간단 설명)
    parser = argparse.ArgumentParser(description="Run NGCF.")
    
    # 모델 저장 경로 지정
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Store model path.')
    
    # 입력 데이터 경로 지정
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    
    # 프로젝트 경로 지정
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    # 사용할 데이터셋 지정 (gowalla, yelp2018, amazon-book 중 하나)
    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    
    # 사전 학습 여부 / 방식 지정 -> 0 = 안함 , -1 = 학습된 임베딩으로 사전학습, 1 = 저장된 모델로 사전 학습
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    
    # 평가 간격 제어 = 정수형 , 매 epoch마다 평가
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    
    # epoch 지정 
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    # 임베딩 벡터 차원 지정
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    
    # layer 출력 크기 설정
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    
    # 배치 크기 설정
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    # 정규화 하이퍼파라미터 설정
    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    
    # 학습률 설정
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    # 모델 유형 설정 (기본 = NGCF)
    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    
    # 인접행렬 유형 설정 (기본 = 정규화)
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')

    # 사용할 GPU ID
    parser.add_argument('--gpu_id', type=int, default=6)

    # 노드 드롭아웃 활성화 여부 설정 (1=활성화)
    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')

    # 노드 드롭아웃 비율 설정
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    
    # 메세지 드롭 아웃 비율 설정
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    # Top-K 추천에 사용될 값 설정
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    # 모델 저장 활성화 여부 ( 0 = 비활성화)
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    # 테스트 방식 ( 기본 = part )
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # 희소성 수준에 따른 성능 보고 활성화 여부
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
