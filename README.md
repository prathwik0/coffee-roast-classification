# Coffee Roast Classification

This project uses a fine-tuned MobileNetV2 model to classify coffee bean roast levels from images. The model can distinguish between four roast levels: Dark, Green, Light, and Medium.

## Project Structure

- `finetune.py`: Script for fine-tuning the MobileNetV2 model on a custom coffee bean dataset.
- `coffee.py`: Script for running the classification model with a Gradio interface.
- `mobilenetv2_finetuned.pth`: The fine-tuned model weights (generated after running `finetune.py`).

## Setup

1. Install the required dependencies:
   ```
   pip install torch torchvision pillow gradio
   ```

2. Ensure you have the dataset in a `train` directory with subfolders for each class.

## Usage

1. Fine-tune the model (optional, if you want to train on your own dataset):
   ```
   python finetune.py
   ```
   This will generate the `mobilenetv2_finetuned.pth` file.

2. Run the classification interface:
   ```
   python coffee.py
   ```
   This will launch a Gradio interface in your default web browser.

3. Use the interface to upload images of coffee beans and see the classification results.

## Example Images

The `examples` folder contains sample images for each roast level that can be used to test the model.

## Note

The dataset used for training is from [Kaggle's Coffee Bean Dataset](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224).
