import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from transformers import AutoModelForImageSegmentation
from PIL import Image
from torchvision import transforms
import io

app = FastAPI()

# 1. 모델 로드 및 장치 설정 (CUDA -> XPU/Arc -> NPU -> CPU)
device = "cpu"
use_npu = False

# 먼저 GPU(CUDA/XPU) 확인
if torch.cuda.is_available():
    device = "cuda"
    print("Device: CUDA")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = "xpu"
    print("Device: Intel XPU (Arc GPU)")
else:
    # IPEX (Intel Extension for PyTorch) 확인
    try:
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            device = "xpu"
            print("Device: Intel XPU (Arc GPU) via IPEX")
    except ImportError:
        pass

# GPU가 감지된 경우 일반적인 방식으로 로드
if device != "cpu":
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)
    model.to(device)
else:
    # GPU가 없으면 NPU 시도 -> 실패 시 CPU
    try:
        from intel_npu_acceleration_library import compile, float16
        from intel_npu_acceleration_library.compiler import CompilerConfig
        print("Device: Intel NPU (Attempting to compile...)")
        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        # NPU 설정을 위한 CompilerConfig 사용
        conf = CompilerConfig(dtype=float16)
        model = compile(model, conf) 
        use_npu = True
        print("Success: Model compiled for NPU")
    except Exception as e:
        print(f"NPU initialization failed or not available: {e}")
        print("Device: CPU")
        device = "cpu"
        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        model.to(device)

model.eval()

# 2. 이미지 전처리 설정
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    # 이미지 읽기
    input_image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    original_size = input_image.size
    
    # 전처리
    input_tensor = transform_image(input_image).unsqueeze(0)
    
    # 장치로 이동 (NPU 사용 시에는 보통 CPU 텐서를 입력으로 받음)
    if not use_npu:
        input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # 마스크 생성 및 합성
    mask = transforms.ToPILImage()(preds[0].squeeze())
    mask = mask.resize(original_size)
    
    # 배경 제거된 이미지 생성 (RGBA)
    input_image.putalpha(mask)
    
    # 결과 반환 (PNG 포맷)
    img_byte_arr = io.BytesIO()
    input_image.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
