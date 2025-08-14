# RAFT Optical Flow 기반 비디오 분류기

이 프로젝트는 RAFT (Recurrent All-Pairs Field Transforms for Optical Flow)를 사용하여 비디오에서 Optical Flow를 추출하고, 이를 ConvGRU 기반의 분류 모델에 입력하여 비디오를 분류하는 전체 파이프라인을 제공합니다.

## 파이프라인 (`run_full_pipeline.sh`)

`run_full_pipeline.sh` 스크립트는 전체 과정을 자동화하며, 3개의 주요 단계로 구성됩니다.

### 1단계: 비디오 준비 (Prepare Videos)

- **스크립트:** `prepare_videos_for_raft.py`
- **동작:** 원본 비디오 파일이 있는 디렉토리에서 비디오들을 읽어와 RAFT 처리에 적합하도록 이름을 변경하고 지정된 폴더에 저장합니다.

### 2단계: Optical Flow 추출 및 시퀀스 생성 (Extract Optical Flow & Trim)

- **스크립트:** `run_of_and_trim.py`
- **동작:** 준비된 비디오들로부터 RAFT 모델을 사용하여 Optical Flow를 추출합니다. 이후, 움직임이 가장 활발한 구간을 찾아 해당 구간의 **전체 프레임** Optical Flow 시퀀스를 `.npz` 파일로 저장합니다. (`CROP_H`, `CROP_W` 등의 인자는 이 단계의 최종 결과물에 영향을 주지 않습니다.)

### 3단계: 분류기 학습 (Train Classifier)

- **스크립트:** `train_classifier.py`
- **동작:** 2단계에서 저장된 Optical Flow 시퀀스를 불러와 학습합니다. `PREPROCESSING_MODE` 설정에 따라, 각 시퀀스 프레임을 리사이즈하거나 특정 위치에서 잘라내는 전처리 과정을 거친 후 모델에 입력합니다. 데이터 증강, 조기 종료, 드롭아웃 등 다양한 기법이 적용되어 과적합을 방지하고 일반화 성능을 높입니다.

---

## 사용 방법

### 1. 설정

`run_full_pipeline.sh` 파일을 열어 상단의 설정 섹션을 필요에 맞게 수정합니다.

#### 주요 설정 변수

- **경로 설정**

  - `INPUT_VIDEO_DIR`: 원본 비디오 파일이 있는 경로.
  - `PREPARED_VIDEO_DIR`: 1단계에서 처리된 비디오가 저장될 경로.
  - `PROCESSED_SEQUENCES_DIR`: 2단계에서 생성된 Optical Flow 시퀀스(`.npz`)가 저장될 경로.
  - `RAFT_MODEL_PATH`: 미리 학습된 RAFT 모델(`.pth`) 파일의 경로.
  - `CLASSIFIER_MODEL_SAVE_PATH`: 3단계에서 학습된 분류기 모델이 저장될 경로.

- **처리 및 학습 설정**
  - `CROP_H`, `CROP_W`: 처리할 프레임의 높이와 너비. 2단계의 ROI 크기와 3단계의 학습 입력 크기로 사용됩니다.
  - `PREPROCESSING_MODE`: 3단계 학습 시 사용할 데이터 전처리 방식을 선택합니다.
    - `'resize'`: 전체 프레임을 `CROP_H`, `CROP_W` 크기로 리사이즈합니다.
    - `'crop'`: 프레임의 특정 부분을 `CROP_H`, `CROP_W` 크기로 잘라냅니다.
  - `CROP_LOCATION`: `PREPROCESSING_MODE`가 `'crop'`일 때 사용됩니다.
    - **특정 위치:** `top-left`, `top-center`, `top-right`, `middle-left`, `center`, `middle-right`, `bottom-left`, `bottom-center`, `bottom-right` 중 하나를 선택합니다.
    - **무작위 위치:** `random`으로 설정 시, 학습 중 9개 위치 중 하나를 무작위로 선택하여 데이터 증강 효과를 줍니다. (권장)
  - `PATIENCE`: 조기 종료(Early Stopping)를 위한 에포크 수. 이 값만큼 검증 성능 향상이 없으면 학습을 중단합니다.
  - `DROPOUT_RATE`: 학습 시 사용할 드롭아웃 비율.
  - `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE` 등: 일반적인 학습 관련 하이퍼파라미터.

### 2. 실행

터미널에서 다음 명령어를 사용하여 파이프라인을 실행합니다.

- **전체 파이프라인 실행:**

  ```bash
  bash run_full_pipeline.sh
  ```

- **특정 단계만 실행:**
  `-s` (시작 단계)와 `-e` (종료 단계) 플래그를 사용하여 원하는 단계만 실행할 수 있습니다.

  예를 들어, 2단계와 3단계만 실행하려면 다음 명령어를 사용합니다.

  ```bash
  bash run_full_pipeline.sh -s 2 -e 3
  ```
