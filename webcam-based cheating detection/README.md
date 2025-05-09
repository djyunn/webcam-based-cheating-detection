# 온라인 시험 부정행위 탐지 시스템

이 프로젝트는 웹캠을 활용하여 온라인 시험 중 부정행위를 감지하는 토이 프로젝트입니다. 시선 추적, 객체 탐지, 머리 방향 추정, 화면 관심 영역(ROI) 모니터링을 통해 응시자의 부정행위를 실시간으로 탐지합니다. 초보자도 접근 가능한 간단한 구현을 목표로 하며, OpenCV, GazeTracking, YOLOv8, Dlib를 사용하여 Python으로 개발되었습니다.

## 프로젝트 개요

이 시스템은 온라인 시험 환경에서 응시자의 부정행위를 감지하기 위해 설계되었습니다. 주요 기능은 다음과 같습니다:
- **시선 추적**: GazeTracking 라이브러리를 사용해 응시자의 시선이 화면 중앙에서 5초 이상 벗어나는지 확인합니다.
- **객체 탐지**: YOLOv8을 활용해 휴대폰, 책 등 부정행위와 관련된 객체를 탐지합니다.
- **머리 방향 추정**: Dlib를 사용해 얼굴 랜드마크를 기반으로 머리 방향(요, 피치, 롤)을 추정하여 응시자가 화면을 향하고 있는지 확인합니다.
- **화면 영역 정의**: 화면을 중앙과 주변 영역으로 나누어 시선과 머리 방향이 정의된 ROI 내에 있는지 판단합니다.

## 상세 요구사항

### 입력
- **웹캠 영상**: 실시간 비디오 스트림 (30fps, 최소 640x480 해상도).
- **환경**: 적절한 조명과 웹캠 품질이 필요.

### 출력
- **부정행위 경고**: 시선, 객체, 머리 방향, ROI 기준으로 부정행위가 감지되면 콘솔 메시지와 화면에 경고 표시.
- **로그 (선택적)**: 타임스탬프와 함께 부정행위 이벤트를 기록.

### 기술 스택
- **OpenCV**: 비디오 캡처 및 프레임 처리 ([OpenCV](https://opencv.org/)).
- **GazeTracking**: 시선 방향 추적 및 깜빡임 감지 ([GazeTracking](https://github.com/antoinelame/GazeTracking)).
- **YOLOv8**: 부정행위 관련 객체 탐지 ([Ultralytics YOLO](https://docs.ultralytics.com/)).
- **Dlib**: 얼굴 랜드마크 탐지 및 머리 방향 추정 ([Dlib](http://dlib.net/)).
- **Python**: 구현 언어.

### 세부 요구사항
- **비디오 캡처**: OpenCV로 실시간 영상 획득.
- **시선 추적**: 시선이 중앙에서 5초 (150프레임, 30fps 기준) 이상 벗어나면 부정행위로 간주.
- **객체 탐지**: YOLOv8으로 휴대폰, 책 등 탐지 (신뢰도 0.5 이상).
- **머리 방향 추정**: Dlib로 얼굴 랜드마크 탐지, 요 각도가 ±30도 이상 벗어나면 부정행위로 간주.
- **화면 영역 정의**: 중앙 ROI (640x480 프레임에서 (100,100)에서 (540,380)) 정의.
- **경고 시스템**: 콘솔 출력 및 화면 메시지 표시.
- **성능**: 프레임당 처리 시간 100ms 이내 목표.
- **프라이버시**: 영상 저장 최소화, 개인정보 보호 준수.

## 전체 아키텍처

| **모듈**                | **설명**                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| 비디오 캡처 모듈        | OpenCV로 웹캠에서 실시간 영상 스트림 획득.                              |
| 시선 추적 모듈          | GazeTracking으로 얼굴과 눈 감지, 시선 방향(왼쪽, 오른쪽, 중앙) 추정.   |
| 객체 탐지 모듈          | YOLOv8으로 휴대폰, 책 등 부정행위 관련 객체 탐지.                      |
| 머리 방향 추정 모듈      | Dlib로 얼굴 랜드마크 탐지, OpenCV로 머리 방향(요, 피치, 롤) 추정.      |
| 화면 영역 정의 모듈      | 화면 중앙 ROI 정의, 시선과 머리 방향이 ROI 내에 있는지 확인.           |
| 부정행위 판단 모듈      | 시선, 객체, 머리 방향, ROI 기준으로 부정행위 판단.                      |
| 경고/로그 모듈          | 부정행위 감지 시 콘솔 출력, 화면 메시지 표시, (선택적) 로그 기록.      |

## 프로세스

1. **웹캠 초기화**: OpenCV로 웹캠 열기, GazeTracking, YOLOv8, Dlib 객체 초기화.
2. **프레임 처리**:
   - GazeTracking으로 시선 방향 분석.
   - YOLOv8으로 객체 탐지 수행.
   - Dlib로 얼굴 랜드마크 탐지 및 머리 방향 추정.
   - 화면 중앙 ROI 정의 및 시선/머리 방향 확인.
3. **부정행위 탐지**:
   - 시선이 중앙이면 카운터 리셋, 비중앙이면 카운터 증가.
   - YOLOv8이 휴대폰, 책 등 탐지 시 경고.
   - 머리 방향(요 각도 ±30도 초과) 벗어나면 경고.
   - 시선 또는 머리 방향이 ROI 밖에 있으면 경고.
4. **경고 생성**:
   - 콘솔에 "부정행위 가능성 감지!" 출력.
   - 화면에 경고 메시지 표시.
   - (선택적) 로그 파일에 기록.
5. **종료**: 'q' 키 입력 시 프로그램 종료.

## 제한사항
- **정확도**: 조명, 웹캠 품질, 얼굴 각도에 따라 성능 달라짐.
- **실시간 처리**: 저사양 장치에서 프레임 드롭 가능성.
- **단순화**: 시선, 머리 방향, ROI 기준이 단순화되어 실제 시험 환경에서는 추가 조정 필요.
- **YOLOv8 성능**: 사전 학습된 모델 사용 시 특정 객체 탐지 정확도가 낮을 수 있음.

## 설치 및 실행 가이드

### 1. 필수 라이브러리 설치
아래 명령어를 사용하여 필요한 라이브러리를 설치합니다:

```bash
# 기본 라이브러리
pip install opencv-python numpy

# 시선 추적 라이브러리
pip install gaze-tracking

# YOLO 객체 탐지
pip install ultralytics

# Dlib 설치 (얼굴 랜드마크 감지용)
pip install dlib
```

### 2. 필요한 모델 파일 다운로드
- **얼굴 랜드마크 모델**:
  - Dlib의 얼굴 랜드마크 예측 모델 다운로드: [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
  - 압축을 풀고 코드와 같은 디렉토리에 저장.
- **YOLO 모델**:
  - 첫 실행 시 `yolov8n.pt`가 자동으로 다운로드됩니다. 인터넷 연결이 필요합니다.

### 3. 시스템 요구 사항
- Python 3.6 이상
- 웹캠이 장착된 컴퓨터
- 적절한 조명 환경 (얼굴이 잘 보이는 조건)
- YOLOv8과 Dlib을 원활하게 실행할 수 있는 적절한 컴퓨팅 성능

### 4. 실행 방법
1. 코드를 `cheating_detection_advanced.py` 파일로 저장합니다.
2. 터미널 또는 명령 프롬프트에서 다음 명령어를 실행합니다:

```bash
python cheating_detection_advanced.py
```

3. 웹캠이 활성화되고 부정행위 탐지 시스템이 시작됩니다.
4. 종료하려면 화면이 활성화된 상태에서 'q' 키를 누르세요.

### 5. 주요 기능
- **시선 추적**: 응시자의 시선이 화면 중앙에서 5초 이상 벗어나면 경고.
- **객체 탐지**: 휴대폰, 책 등 부정행위 관련 객체 감지.
- **머리 방향 추정**: 머리가 화면을 향하고 있는지 확인.
- **관심 영역(ROI) 모니터링**: 시선과 머리 방향이 정의된 화면 영역 내에 있는지 확인.

### 6. 문제 해결
- **웹캠 인식 오류**: 장치 관리자에서 웹캠이 정상 작동하는지 확인하세요.
- **라이브러리 설치 오류**: Python 버전과 호환되는 라이브러리 버전을 확인하세요.
- **Dlib 설치 문제**: Windows의 경우 Visual Studio C++ Build Tools가 필요할 수 있습니다 ([PyImageSearch](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)).
- **성능 저하**: 저사양 장치에서는 YOLOv8n 대신 더 가벼운 모델을 사용하거나, 프레임 처리 간격을 늘리는 것을 고려하세요.

### 7. 코드 사용자 정의
- **부정행위 임계값 조정**: `threshold` 값을 변경하여 시선 탐지 민감도 조정 (기본값 150 = 5초).
- **ROI 영역 조정**: `roi_x`, `roi_y`, `roi_w`, `roi_h` 값을 조정하여 관심 영역 크기 변경.
- **객체 탐지 추가**: YOLOv8의 COCO 데이터셋 클래스 ID를 추가하여 다른 객체 탐지 가능 (예: 키보드, 노트북).

## Key Citations
- [GazeTracking Library for Eye Tracking](https://github.com/antoinelame/GazeTracking)
- [Ultralytics YOLO Documentation for Object Detection](https://docs.ultralytics.com/)
- [Dlib Library for Machine Learning and Face Detection](http://dlib.net/)
- [LearnOpenCV Head Pose Estimation Tutorial](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
- [PyImageSearch Dlib Installation Guide](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)
- [Detection of Malpractice in E-exams by Head Pose and Gaze Estimation](https://www.semanticscholar.org/paper/Detection-of-Malpractice-in-E-exams-by-Head-Pose-Indi-Pritham/552358015d28db0178372b1a51e956f99b19e657)
- [GitHub Test-Cheating-Detection Project](https://github.com/akshaysatyam2/Test-Cheating-Detection)