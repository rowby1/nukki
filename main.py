import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from transformers import AutoModelForImageSegmentation
from PIL import Image
from torchvision import transforms
import io

app = FastAPI()

# 1. 모델 로드 (GPU 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", 
    trust_remote_code=True
)
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
    
    # 전처리 및 추론
    input_tensor = transform_image(input_image).unsqueeze(0).to(device)
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