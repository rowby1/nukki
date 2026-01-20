# BiRefNet 배경 제거 API (Nukki Server)

BiRefNet 모델을 활용하여 이미지의 배경을 고품질로 제거하는 간단한 FastAPI 기반의 API 서버입니다. 이 프로젝트는 누구나 쉽게 고성능 배경 제거 기능을 API로 사용할 수 있도록 공개되었습니다.

## 주요 기능 (Features)

- **고품질 배경 제거**: 최신 BiRefNet 모델을 사용하여 머리카락 등 미세한 부분까지 정교하게 배경을 제거합니다.
- **REST API 제공**: FastAPI로 구현되어 있어 웹, 앱 등 다양한 클라이언트에서 쉽게 연동 가능합니다.
- **GPU 가속 지원**: CUDA가 지원되는 환경에서 고속 추론이 가능합니다 (CPU 모드도 자동 지원).
- **Docker 지원**: 컨테이너 환경에서 손쉽게 배포 및 실행이 가능합니다.

## 설치 및 실행 (Installation & Usage)

### 사전 요구 사항 (Prerequisites)

- Python 3.10 이상
- NVIDIA GPU + CUDA (권장, 없을 시 CPU로 동작)

### 1. 로컬 환경에서 실행

**설치**

```bash
# 저장소 클론
git clone [REPOSITORY_URL]
cd nukki

# 의존성 패키지 설치
pip install -r requirements.txt
```

**실행**

```bash
# 서버 시작
python main.py
```
서버는 기본적으로 `0.0.0.0:8001`에서 실행됩니다.

### 2. Docker로 실행

Docker가 설치되어 있다면 환경 설정 고민 없이 바로 실행할 수 있습니다.

```bash
# Docker 이미지 빌드
docker build -t nukki-server .

# 컨테이너 실행 (GPU 사용 시 --gpus all 옵션 필요)
docker run --gpus all -p 8001:8001 nukki-server
```

## API 사용법 (API Documentation)

서버가 실행 중일 때 `http://localhost:8001/docs` 에 접속하면 Swagger UI를 통해 API를 직접 테스트해볼 수 있습니다.

### 배경 제거 요청

- **Endpoint**: `POST /remove-bg`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: 배경을 제거할 이미지 파일 (Binary)

### Python 요청 예시

```python
import requests

url = "http://localhost:8001/remove-bg"
file_path = "input_image.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open("output_no_bg.png", "wb") as f:
        f.write(response.content)
    print("배경 제거 완료: output_no_bg.png")
else:
    print("에러 발생:", response.text)
```

## 라이선스 (License)

이 프로젝트는 **MIT 라이선스**로 배포됩니다.
또한, 사용된 **BiRefNet** 모델 역시 **MIT 라이선스**를 따릅니다. 누구나 자유롭게 사용 및 수정이 가능합니다.
