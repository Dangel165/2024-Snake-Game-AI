# 🐍 Snake Game AI with Deep Q-Learning

PyTorch를 사용한 강화학습 기반 스네이크 게임 AI입니다.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 목차
- [특징](#-특징)
- [사용법](#-사용법)
- [작동 원리](#-작동-원리)
- [프로젝트 구조](#-프로젝트-구조)
- [학습 결과](#-학습-결과)

## ✨ 특징

- 🧠 **Deep Q-Learning**: PyTorch 기반 신경망으로 학습
- 🎮 **실시간 학습**: 게임을 플레이하면서 실시간으로 학습
- 📊 **시각화**: 학습 진행 상황을 그래프로 실시간 확인
- 💾 **모델 저장**: 최고 기록 달성 시 자동으로 모델 저장
- 🚀 **빠른 학습**: 100-200 게임 후 눈에 띄는 성능 향상

## 🎬 데모

```
Game 1: Score 1, Record: 1
Game 50: Score 8, Record: 12
Game 100: Score 15, Record: 20
Game 200: Score 25, Record: 35
```

## 🚀 사용법

### AI 학습 시작
```bash
python agent.py
```

### 게임만 플레이 (AI 없이)
```bash
python snake_game.py
```

### 학습 중단
- `Ctrl + C` 또는 게임 창 닫기

## 🧠 작동 원리

### 1. 상태 (State)
AI는 11개의 입력을 받습니다:
- 위험 감지 (직진, 우회전, 좌회전)
- 현재 이동 방향 (상, 하, 좌, 우)
- 음식 위치 (상, 하, 좌, 우)

### 2. 행동 (Action)
3가지 행동 중 선택:
- `[1, 0, 0]`: 직진
- `[0, 1, 0]`: 우회전
- `[0, 0, 1]`: 좌회전

### 3. 보상 (Reward)
- 음식 먹기: +10
- 게임 오버: -10
- 그 외: 0

### 4. 신경망 구조
```
Input Layer (11) → Hidden Layer (256) → Output Layer (3)
```

### 5. 학습 알고리즘
- **DQN (Deep Q-Network)**
- **Experience Replay**: 과거 경험을 저장하고 재학습
- **Epsilon-Greedy**: 탐험과 활용의 균형

## 📁 프로젝트 구조

```
snake-ai/
│
├── agent.py              # AI 에이전트 및 학습 로직
├── model.py              # PyTorch 신경망 모델
├── snake_game.py         # 스네이크 게임 엔진
├── requirements.txt      # 필요한 패키지
│
└── model/
    └── model.pth         # 학습된 모델 (자동 생성)
```

## 📊 학습 결과

### 학습 단계
1. **초기 (0-80 게임)**: 랜덤 탐험, 점수 0-5
2. **학습 (80-200 게임)**: 패턴 학습 시작, 점수 5-15
3. **숙련 (200+ 게임)**: 안정적인 플레이, 점수 15-30+

### 그래프 해석
- **파란선**: 각 게임의 점수
- **주황선**: 평균 점수 (학습 진행도)

## ⚙️ 커스터마이징

### 하이퍼파라미터 조정 (agent.py)
```python
MAX_MEMORY = 100_000      # 메모리 크기
BATCH_SIZE = 1000         # 배치 크기
LR = 0.001                # 학습률
```

### 게임 속도 조정 (snake_game.py)
```python
SPEED = 40  # 높을수록 빠름 (기본: 40)
```

### 신경망 크기 조정 (agent.py)
```python
self.model = Linear_QNet(11, 256, 3)  # (입력, 은닉층, 출력)
```

## 🎯 학습 팁

1. **인내심**: 100-200 게임은 돌려야 확실한 효과
2. **모델 저장**: `model/model.pth`에 최고 기록 모델 저장됨
3. **재학습**: 기존 모델이 있으면 이어서 학습 가능
4. **실험**: 하이퍼파라미터를 바꿔가며 실험해보세요

## 🐛 문제 해결

### PyTorch 설치 오류
```bash
# CPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 버전 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 그래프가 안 보임
```bash
pip install matplotlib ipython
```

## 📚 학습 자료

- [Deep Q-Learning 논문](https://arxiv.org/abs/1312.5602)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [강화학습 기초](https://www.youtube.com/watch?v=2pWv7GOvuf0)

## 📄 라이선스

MIT License 

## 👨‍💻 개발자

**Dangel**
