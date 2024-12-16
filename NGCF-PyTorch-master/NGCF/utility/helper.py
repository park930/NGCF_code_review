'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re

# 지정된 file_src에서 파일을 읽기 모드로 열어, orig_file 생성
def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

# dir_path에서 디렉토리 부분만 추출, 디렉토리 없으면 새로 생성
def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

# unicode_str을 아스키 형식으로 인코딩
def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

# 정규식 써서 문자열 inputString에 숫자가 포함되어 있는지 확인
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

# 입력 문자열 char의 각 문자를 하나씩 순회해서 inputString에서 각 문자를 제거함
def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

# dict인 x,y를 병합해 새로운 dict Z를 만듦
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

# 
def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy: 올바른 정렬 기준인지
    # expected_order가 acc or dec 중 하나인지 확인
    assert expected_order in ['acc', 'dec']

    # log값 >= best값 : stopping_step = 0 , best로 설정
    # 성능 개선되었으면, 조기 중단 카운터 초기화 + 최고 성능값 갱신
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    
    # 성능 개선 X = 조기 중단 카운터 + 1
    else:
        stopping_step += 1

    # 조기 중단 카운터가 최대값(flag_step)에 도달했는지 확인
    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    
    # 학습 계속 가능
    else:
        should_stop = False
    
    # 현재 최고 성능 값, 조기 중단 카운터, 중단 여부 플래그를 반환 
    return best_value, stopping_step, should_stop
