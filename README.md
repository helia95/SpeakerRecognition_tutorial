# SpeakerRecognition_tutorial

LG전자 서초R&D캠퍼스 강의 - 화자인식 튜토리얼

## 필요 환경
python 3.5+  
pytorch 1.0.0  
pandas 0.23.4  
numpy 1.13.3  
pickle 4.0  
matplotlib 2.1.0  

## DB
아래의 과제를 통해 수집된 DB를 이용 
- No. 10063424, '실내용 음성대화 로봇을 위한 원거리 음성인식 기술 및 멀티 태스크 대화처리 기술 개발'
- 2차년도 수집 DB

DB 정보
- SNS단문 낭독 음성 DB (ETRI 낭독체)
- 원거리, 무잡음 음성, 1m 거리, 0도 방향, 16kHz, 16bits  

위의 DB를 이용하여 추출한 log mel filterbank energy feature를 업로드하였습니다.

### * 훈련 DB
24000개 파일, 240개 폴더(240명 화자)  
feat_logfbank_nfilt40 - train

### * 등록 및 테스트 DB
20개 파일, 10개 폴더(10명 화자)  
feat_logfbank_nfilt40 - test

## 사용법
### 1. 훈련
train.py  

### 2. 등록
enroll.py  

### 3. 테스트
verification.py - 화자검증  
identification.py - 화자식별  



## 문의
dudans@kaist.ac.kr
