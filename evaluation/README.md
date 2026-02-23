# Evaluation for ReMOVE+

*ReMOVE+* is a modified reference-free metric designed to address critical limitations of the original *ReMOVE* metric in object-effect removal tasks: the original ReMOVE only judges whether the object erasure region of the output result is harmonious with the output background (leading to high scores even when the background is severely altered) and cannot effectively evaluate the removal of object effects. To solve these issues, ReMOVE+ assesses removal success by measuring the consistency between the object-effect region of the output result and the area outside the object-effect region of the input image, making it suitable for OBER-Wild dataset evaluation.

## 1. Clone the Base Repository
First, clone the official ReMOVE repository and set up the base environment:
```bash
git clone https://github.com/chandrasekaraditya/ReMOVE.git 
cd ReMOVE
```

## 2. Environment Requirements

### Install Dependencies
1. Install PyTorch and TorchVision with CUDA support
2. Install SAM:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

## 3. Download Pre-trained Model
Download the ViT-H model (used by default in ReMOVE):
```bash
# Create models directory if not exists
mkdir -p models
# Download model using the url in models/url.txt
wget -i models/url.txt -P models/
```

## 4. ReMOVE+ Evaluation

### Step 1: Navigate to Evaluation Folder
Ensure you are in the `evaluation` directory:
```bash
cd ..
```

### Step 2: Run ReMOVE+
#### Basic Command
```bash
python remove_plus.py \
  -ind OBER-Wild inputs \
  -rd OBER-Wild model results \
  -md OBER-Wild object-effect-mask \
  --save_csv [your_save_path]/evaluation_results.csv
```
