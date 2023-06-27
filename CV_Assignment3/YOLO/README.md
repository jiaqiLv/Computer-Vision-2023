# Instant Noodles Detection

## How to run

1. Unzip the `yolov5.tar` and open the project.

2. Create the virtual environment and install the required dependency packages

3. Train the model (option)

   ```
   python train.py --data instant_noodles.yaml --weights yolov5x.pt --img 640
   ```

   > tips: If you want to train the model by yourself, you need to get the labeled dataset. Please connect with me to get it.

4. Detect using pre-trained model

   ```
   python detect.py --weights ./runs/train/exp2/weights/best.pt --source testvideo.mp4
   ```

## Tips

- You can get the training result from `result` folder
- `sort.py` is used to get the sorted `pictures` and `label files`