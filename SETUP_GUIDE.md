# IT-RAP 실험 환경 셋팅 가이드 (완전 신규 환경)

> **대상**: Random / Greedy baseline 포함 전체 inference 실험을 새 서버에서 처음부터 실행하는 경우

---

## 0. 전제 조건

| 항목 | 권장 사양 |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| GPU | NVIDIA GPU (VRAM 8GB 이상 권장) |
| CUDA | 12.1 또는 11.8 |
| Python | 3.10 ~ 3.12 |
| Conda | Miniconda 또는 Anaconda |
| 디스크 여유 공간 | 30GB 이상 (CelebA ~2GB, MAAD-Face ~20GB, 모델 ~1GB) |

---

## 1. Conda 가상환경 생성

```bash
# Python 3.11 환경 생성 (권장)
conda create -n itrap python=3.11 -y
conda activate itrap
```

---

## 2. 레포지토리 클론 (rebuttal 브랜치)

```bash
git clone -b rebuttal https://github.com/tedrudwls/IT-RAP.git
cd IT-RAP
```

---

## 3. PyTorch 설치 (CUDA 버전에 맞게 선택)

> `nvidia-smi` 명령으로 CUDA 버전을 먼저 확인하세요.

**CUDA 12.1 사용 시 (권장)**
```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 사용 시**
```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (테스트 목적)**
```bash
pip install torch==2.2.2 torchvision==0.17.2
```

---

## 4. 나머지 패키지 설치

```bash
pip install \
    numpy==1.26.4 \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scikit-image \
    PyWavelets \
    "torchmetrics>=1.0.0" \
    opencv-python \
    Pillow \
    torch-dct \
    optuna \
    h5py \
    face-alignment \
    python-dotenv \
    neptune \
    lmdb \
    dlib \
    sympy

# CLIP (OpenAI)
pip install git+https://github.com/openai/CLIP.git

# EdgeFace (feature extractor, torch.hub으로 자동 다운로드되지만 의존성 사전 설치)
pip install ellzaf-ml
```

> **참고**: `dlib` 설치 시 cmake가 필요합니다. 오류 발생 시 `sudo apt-get install cmake`를 먼저 실행하세요.

---

## 5. 데이터셋 다운로드

### 5-1. CelebA 데이터셋

```bash
# IT-RAP 레포 내 download.sh 스크립트 사용
bash download.sh celeba
```

다운로드 후 디렉토리 구조:
```
IT-RAP/
└── data/
    └── celeba/
        ├── images/          ← 얼굴 이미지 (202,599장)
        └── list_attr_celeba.txt  ← 속성 레이블 파일
```

### 5-2. MAAD-Face 데이터셋

[MAAD-Face 공식 레포](https://github.com/pterhoer/MAAD-Face)의 안내에 따라 다운로드한 후, 아래 구조로 배치합니다.

```
IT-RAP/
└── MAAD-Face/
    ├── data/
    │   └── train/           ← 얼굴 이미지들
    └── MAAD_Face_filtered.csv  ← 속성 레이블 CSV
```

> `MAAD_Face_filtered.csv`는 `Filename` 컬럼과 속성 컬럼들(`Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Male`, `Young` 등)로 구성된 CSV 파일이어야 합니다.

---

## 6. 사전학습 모델 다운로드

### 6-1. StarGAN 사전학습 모델 (CelebA 256×256)

```bash
bash download.sh pretrained-celeba-256x256
```

다운로드 후 경로:
```
IT-RAP/checkpoints/models/200000-G.ckpt   ← StarGAN Generator
IT-RAP/checkpoints/models/200000-D.ckpt   ← StarGAN Discriminator
```

### 6-2. AttGAN 사전학습 모델

[Google Drive 링크](https://drive.google.com/drive/folders/1JMQ-gtI4rmdkmnPSIHw0cMGRBRa2Hw1z?usp=sharing)에서 `256_shortcut1_inject1_none_hq` 폴더를 다운로드합니다.

다운로드 후 아래 경로에 배치:
```
IT-RAP/attgan/256_shortcut1_inject1_none_hq/checkpoint/weights.199.pth
```

디렉토리가 없으면 직접 생성:
```bash
mkdir -p attgan/256_shortcut1_inject1_none_hq/checkpoint/
# weights.199.pth 파일을 위 경로로 이동
```

### 6-3. IT-RAP RL 체크포인트 (학습된 Rainbow DQN)

학습된 `final_rainbow_dqn.pth`를 아래 경로에 배치합니다:
```
IT-RAP/checkpoints/models/final_rainbow_dqn.pth
```

**만약 학습된 체크포인트가 없다면 (Random/Greedy baseline만 실행하는 경우)**:
Random 및 Greedy policy는 RL 체크포인트를 로드하지 않으므로 초기화된 더미 파일만 있어도 됩니다.

```bash
# 더미 초기화 파일 생성 (Random/Greedy 실험 전용)
python initialize_dqn.py
```

---

## 7. Neptune 설정 (실험 로깅)

`attgan_main.py` / `stargan_main.py`는 Neptune을 사용합니다. 프로젝트 루트에 `.env` 파일을 생성합니다:

```bash
cat > .env << 'EOF'
NEPTUNE_PROJECT=your-workspace/your-project-name
NEPTUNE_API_TOKEN=your-neptune-api-token
EOF
```

> Neptune 계정이 없거나 로깅이 불필요한 경우, `stargan_main.py` / `attgan_main.py`의 Neptune 초기화 부분을 주석 처리하거나 오프라인 모드로 변경해야 합니다.

---

## 8. 최종 디렉토리 구조 확인

모든 준비가 완료된 후 디렉토리 구조:

```
IT-RAP/
├── data/celeba/
│   ├── images/
│   └── list_attr_celeba.txt
├── MAAD-Face/
│   ├── data/train/
│   └── MAAD_Face_filtered.csv
├── attgan/
│   └── 256_shortcut1_inject1_none_hq/
│       └── checkpoint/
│           └── weights.199.pth
├── checkpoints/models/
│   ├── 200000-G.ckpt          ← StarGAN Generator
│   ├── 200000-D.ckpt          ← StarGAN Discriminator
│   └── final_rainbow_dqn.pth  ← IT-RAP RL 체크포인트
├── .env                        ← Neptune API 키
└── (코드 파일들...)
```

---

## 9. 실험 실행

### 9-1. RL (기존 IT-RAP) Inference

```bash
# CelebA + AttGAN
python "[CelebA] infer_entry_point.py"

# MAADFace + StarGAN
python "[MAADFace] infer_entry_point.py"
```

### 9-2. Random Baseline Inference (신규)

```bash
python "[CelebA_StarGAN] random_infer_entry_point.py"
python "[CelebA_AttGAN] random_infer_entry_point.py"
python "[MAADFace_StarGAN] random_infer_entry_point.py"
python "[MAADFace_AttGAN] random_infer_entry_point.py"
```

### 9-3. Greedy Baseline Inference (신규)

```bash
python "[CelebA_StarGAN] greedy_infer_entry_point.py"
python "[CelebA_AttGAN] greedy_infer_entry_point.py"
python "[MAADFace_StarGAN] greedy_infer_entry_point.py"
python "[MAADFace_AttGAN] greedy_infer_entry_point.py"
```

결과는 각각 `result_random_{dataset}_{model}/`, `result_greedy_{dataset}_{model}/` 디렉토리에 저장됩니다.

---

## 10. 자주 발생하는 오류 및 해결법

| 오류 | 원인 | 해결법 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch_dct'` | torch-dct 미설치 | `pip install torch-dct` |
| `ModuleNotFoundError: No module named 'ellzaf_ml'` | ellzaf-ml 미설치 | `pip install ellzaf-ml` |
| `FileNotFoundError: weights.199.pth` | AttGAN 체크포인트 경로 오류 | `attgan/256_shortcut1_inject1_none_hq/checkpoint/` 경로 확인 |
| `FileNotFoundError: final_rainbow_dqn.pth` | RL 체크포인트 없음 | `python initialize_dqn.py` 실행 |
| `neptune.exceptions.NeptuneInvalidApiTokenException` | .env 파일 미설정 | `.env` 파일에 API 토큰 입력 |
| `CUDA out of memory` | GPU 메모리 부족 | `--batch_size 1` 확인, 다른 프로세스 종료 |
| `dlib` 설치 오류 | cmake 미설치 | `sudo apt-get install cmake libopenblas-dev` |
| EdgeFace 다운로드 실패 | GitHub hub 연결 오류 | 인터넷 연결 확인, `trust_repo` 관련 torch.hub 캐시 삭제 후 재시도 |
