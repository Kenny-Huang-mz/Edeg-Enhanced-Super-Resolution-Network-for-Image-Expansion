# Edge-Guided Laplacian Enhanced SRCNN

**A modified implementation of SRCNN that integrates edge-aware Laplacian enhancement and Sobel-guided feature fusion to improve visual clarity in super-resolved images..**

## 📌 Overview

This project proposes an improved version of the SRCNN (Super-Resolution Convolutional Neural Network) by integrating:

-  **Edge-Guided Laplacian Filtering** to amplify structural details before feeding into the network.
-  **Sobel Feature Concatenation** at feature level to reinforce spatial sharpness.
-  An end-to-end trainable PyTorch pipeline

The model is especially effective at enhancing **structural clarity** (edges and contours) while maintaining global smoothness in upsampled grayscale images.

## 📁 Directory Structure
```
├── model.py
├── train.py
├── inference_x2.py
├── config.py
├── dataset.py
├── imgproc.py
├── data/
│   ├── T91/SRCNN/train_binary/   # We only use this grey image set
│   └── Set5/GTmod12/             # Benchmark for validation
│   └── README.md
├── results/                      # Evaluation outputs and pretrained weights
├── samples/                      # Saved models during training
├── my_images/                    # The image that you want to modify
requirements.txt
```

## 🚀 Installation
```
git clone https://github.com/yourusername/edge-guided-srcnn.git
cd edge-guided-srcnn
conda create -n SRCNN python=3.9
conda activate SRCNN
pip install -r requirements.txt
```

## Download datasets
Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc. But we only use the T91
- [Google Driver]([https://your-google-drive-link.com](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8))
- [Baidu Driver]([https://your-baidu-drive-link.com](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot))

## Train your model
```
# Edit config.py to set mode="train"
python train.py --dataroot ./data/T91 --upscale_factor 2
```

## Run inference on your own image
```
# Edit config.py to set mode="test"
python inference_x2.py \
  --inputs_path my_images/camera.png \
  --output_path my_results/camera_sr.png \
  --weights_path results/SRCNN_x2_binary_edge_enhance/best.pth.tar
```
