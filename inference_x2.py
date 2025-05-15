# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import cv2
import numpy as np
import torch
import config
import imgproc
from model import SRCNN


def main(args):
    # Initialize the model
    model = SRCNN()
    model = model.to(memory_format=torch.channels_last, device=config.device)
    print("Build SRCNN model successfully.")

    # Load the SRCNN model weights
    checkpoint = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRCNN model weights `{args.weights_path}` successfully.")

    # Start the model in evaluation mode
    model.eval()

    # Read the input image
    lr_image = cv2.imread(args.inputs_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # ⬆️ Step 1: Bicubic upscale
    upscale_factor = config.upscale_factor
    h, w = lr_image.shape[:2]
    lr_image = cv2.resize(lr_image, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)

    # Step 2: Convert to Y channel
    lr_y_image = imgproc.bgr2ycbcr(lr_image, True)
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, False)
    _, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)

    # Step 3: Convert Y to tensor and move to device
    lr_y_tensor = imgproc.image2tensor(lr_y_image, False, False).unsqueeze(0).to(
        config.device, memory_format=torch.channels_last, non_blocking=True
    )

    # Step 4: Run SRCNN on Y channel
    with torch.no_grad():
        sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

    # Step 5: Convert back to image
    sr_y_image = imgproc.tensor2image(sr_y_tensor, False, False).astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image, lr_cb_image, lr_cr_image])
    sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)

    # Save final SR image
    cv2.imwrite(args.output_path, sr_image * 255.0)
    print(f"SR image saved to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SRCNN on an upscaled image.")
    parser.add_argument("--inputs_path", type=str, help="Path to input image.")
    parser.add_argument("--output_path", type=str, help="Path to save output image.")
    parser.add_argument("--weights_path", type=str, help="Path to trained model weights.")
    args = parser.parse_args()

    main(args)
