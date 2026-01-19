# NVIDIA CUDA 지원 파이토치 이미지 사용
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY main.py .

# API 포트 노출
EXPOSE 8001

# 서버 실행
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8001"]