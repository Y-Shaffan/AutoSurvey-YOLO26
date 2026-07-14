import torch
from ultralytics import YOLO

# device = torch.device('xpu' if torch.cuda.is_available() or torch.xpu.is_available() else 'cpu') # Verify if Intel GPU is available, else use CPU
torch.set_num_threads(4) # Use this for CPU
# print(f"Using device: {device}")

model = YOLO("yolo26n.pt")


model.train(
        data='training13_07_2026.yaml', 
        epochs=100, 
        imgsz=640,
        # device="str(device)" # Forces Ultralytics to use the 'xpu' device
        device = "cpu",
        workers=2 # Use this if using CPU
    )
