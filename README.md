## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Models](#models)
- [Contributors](#contributors)

## Introduction

This project was conducted from June 2022 to March 2023 on the development of an anomaly detection and prediction model for time-series data collected in real-time from manufacturing process equipment.

- **프로젝트 목표**: 제조 공정 설비에서 실시간으로 수집되는 시계열 데이터를 기반으로 이상징후(Anomaly) 탐지 및 예측을 수행하고, 대시보드로 시각화/모니터링합니다.
- **핵심 구성**: 프론트엔드 대시보드(React/Vite), 백엔드 API(FastAPI), 모델링 코드(TCAD/MSCVAE/통합 모델) 및 시각화 스크립트.

#### Demo

- **Demo site**: `https://ai.studio/apps/drive/1QIWmsE6sn0cEUbPLvf6WN-OY4X8plDAa`

#### Repository Structure

아래는 주요 폴더/파일을 중심으로 한 구조 요약입니다.

```text
.
├─ packages/
│  ├─ backend/                # FastAPI 서버 (실시간 점수/로그/모델 목록)
│
├─ components/                # 대시보드 UI 위젯(차트/로그/요약 등)
│
├─ utils/                     # 프론트용 API 클라이언트/엔드포인트 설정
│
├─ modeling/                  # 모델 학습/평가/시각화 스크립트(로컬 실행용)
│  ├─ data/                   # 학습/시각화용 CSV (sensor_data_rms2_fixed.csv)
│  ├─ tcad/                   # TCAD 모델 학습/평가/시각화
│  ├─ mscvae/                 # MSCVAE 모델 학습/평가/시각화
│  └─ integrated/             # TCAD + MSCVAE 특징을 결합한 통합 모델(실험)
│
├─ App.tsx                    # 루트 대시보드
└─ ...
```

## Quick Start

### Local Development (Without Docker)

#### 1. Setup Environment Variables

Create a `.env` file in the project root:

```bash
# GCP Authentication (required)
GCP_PROJECT_ID=your-project-id
GCP_PRIVATE_KEY_ID=your-private-key-id
GCP_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n
GCP_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
GCP_CLIENT_ID=your-client-id

# GCS Configuration
GCS_BUCKET=your-bucket-name
DATASET_GCS_PATH=data/sensor_data_rms2_fixed.csv
MDR_MODEL_GCS_PATH=models/mdr_model.pt

# Local development: no API prefix (direct access)
USE_API_PREFIX=false
```

#### 2. Start Backend (FastAPI)

```bash
cd packages/backend
pip install -r requirements.txt
python main.py
```

Backend will run at `http://localhost:8000`

#### 3. Start Frontend (Vite)

```bash
npm install
npm run dev
```

Frontend will run at `http://localhost:5173` and automatically connect to backend at `http://localhost:8000`

### Production Deployment (Docker)

```bash
docker-compose up --build
```

This will:
- Build frontend with Nginx proxy configuration
- Run backend with `/api` prefix
- Serve everything on port 80

#### API (Backend)

- **Health**
  - `GET /health`
- **System logs**
  - `GET /logs?limit=100` (옵션: `cursor`, `wrap`)
- **Realtime anomaly scores**
  - (프론트 기본값) `GET /realtime/scores`
  - (백엔드/문서에 포함된 변형) `GET /realtime/data`, `POST /realtime/data`
- **Models list**
  - `GET /models/list`

## Models

#### 1) TCAD (`modeling/tcad`)

- **핵심 아이디어**
  - Transformer 기반 **전역(Global) 컨텍스트 인코더**
  - ResNet(1D Conv) 기반 **지역(Local) 패턴 인코더**
  - \(z_1\)과 \(z_2\)의 표현 불일치(discrepancy) + 재구성 오차를 통해 이상 점수를 산출
- **주요 파일**
  - `modeling/tcad/model.py`: TCAD 아키텍처
  - `modeling/tcad/train.py`, `eval.py`, `visualize.py`

#### 2) MSCVAE (`modeling/mscvae`)

- **핵심 아이디어**
  - 시계열의 각 시점에서 센서 간 상관 구조를 나타내는 attribute matrix(outer product)를 구성
  - VAE + temporal 모델링(ConvLSTM)으로 상관 구조의 재구성 난이도를 이상 점수로 활용
- **주요 파일**
  - `modeling/mscvae/model.py`: MSCVAE 아키텍처
  - `modeling/mscvae/utils.py`: `attribute_matrix`
  - `modeling/mscvae/train.py`, `eval.py`, `visualize.py`

#### 3) IntegratedFusionAD (`modeling/integrated`) — TCAD + MSCVAE 통합(실험)

TCAD(전역/지역)과 MSCVAE(상관 구조)의 장점을 결합한 다중 스트림 통합 인코더-디코더 모델입니다.

- **3개 인코더 스트림**
  - **Global**: Transformer encoder → $(Z_1)$
  - **Local**: ResNet(1D Conv) encoder → $(Z_2)$
  - **Correlation**: attribute matrix 기반 CNN encoder → $(Z_3)$
- **융합(Fusion)**
  - $(Z_1, Z_2, Z_3)$를 attention/gating으로 가중합하여 단일 latent $(Z)$로 통합
- **멀티태스크 디코딩**
  - 원본 시계열 재구성 $( \hat X )$
  - (시퀀스 평균) attribute matrix 재구성 $( \hat M )$
- **이상 점수(Anomaly score)**
  - raw reconstruction error + correlation reconstruction error + latent discrepancy
  - GLAD 개념처럼 “전역/지역/구조 표현 간의 차이”를 조기 이상징후 신호로 활용
- **주요 파일**
  - `modeling/integrated/model.py`, `loss.py`, `train.py`, `eval.py`, `visualize.py`, `dataset.py`

## Contributors

Here is the list of contributors who participated in this project.

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chosungsu"><img src="https://avatars.githubusercontent.com/u/48382347?v=4?s=100" width="100px;" alt="chosungsu"/><br /><sub><b>chosungsu</b></sub></a><br /><a href="https://github.com/ChocoPytoch/BISTelligence/commits?author=chosungsu" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kmw4097"><img src="https://avatars.githubusercontent.com/u/98750892?v=4?s=100" width="100px;" alt="kmw4097"/><br /><sub><b>kmw4097</b></sub></a><br /><a href="https://github.com/ChocoPytoch/BISTelligence/commits?author=kmw4097" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dbnub"><img src="https://avatars.githubusercontent.com/u/99518647?v=4?s=100" width="100px;" alt="dbnub"/><br /><sub><b>dbnub</b></sub></a><br /><a href="https://github.com/ChocoPytoch/BISTelligence/commits?author=dbnub" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/choiyongwoo"><img src="https://avatars.githubusercontent.com/u/50268222?v=4?s=100" width="100px;" alt="choiyongwoo"/><br /><sub><b>choiyongwoo</b></sub></a><br /><a href="https://github.com/ChocoPytoch/BISTelligence/commits?author=choiyongwoo" title="Commits">📖</a> </td>
    </tr>
  </tbody>
</table>
