# Robotic Pallet Detail Detection

This a learning project with **PyTorch** and **YOLOv8/TensorRT** to detect overlapping mechanical components on pallets using a live camera feed (I'll also had to use my phone for now)
Training stages: singles → groups → overlaps
Training with only 20-50 images in dataset for each round.
I also Transfered leraning on each step and presompiled it for RTX3080

# Install dependencies
```sh
pip install -r requirements.txt
```
# HOw to Run
(camera ON)
1. Collect curiculum dataset. (I did it in 3 runs for singes/groups/overlaps)
```sh
python camera_pipe/capture.py
```
2. Train the NN weights
```sh
python train.py
```
3. Compile for NVIDIA <span style='color: orange'> **if you have one!!**</span>
```sh
yolo export model=stick_detector.pt format=engine device-0 half=True
```
# Finally: Run the model with live camera
```sh
python RT_tracker.py
```
#
#
## Pipeline
(Batch Inference for now)
1. I'll be **auto-Labeling:** extracting initial bounding boxes using my Scikit-image/CV2 pipeline I used before
2. annotation tools - adjust boexes
3. Fine-tunnig a pre-trained YOLO model on the RTX 3080.
4. Feed coordinates to the robot's solvers

# Testing with precompiled model for RTX-3080:
### my_model.pt → blueprint file called ONNX (.onnx) → my_optimized_model.engine
```sh
# Python bindings
pip install tensorrt tensorrt-cu12==[some-version]
# C++ for main execution runtime
pip install tensorrt-cu12-libs
# Dispatecher for Orchestration and Executaion
pip install tensorrt-dispatch-cu12

# compile and export an engine
yolo export model=stick_detector.pt format=engine device=0
```

# Notes & Throubles:
## My notes:
- **Environment Manager:** Conda (`yolo_bot`)
- **Core Frameworks:** PyTorch on CUDA 12.1, ultralytics for yolo for YOLO

```sh
conda create --name yolo_bot python=3.10 -y
conda activate yolo_bot
conda install -c nvidia cuda-toolkit=12.1 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install ultralytics
```

## my throubles
### <span style='color: yellow'> wrong masking!</span> :( - fix it!
### NVIDIAs packages (tensorrt...) are rather large and may fail. Try again :(