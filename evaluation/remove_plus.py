import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
import argparse
import csv

remove_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ReMOVE"))
if remove_dir not in sys.path:
    sys.path.insert(0, remove_dir)

from crop import find_smallest_bounding_square, draw_bb
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor

import warnings
warnings.filterwarnings("ignore")

def merge_images(input_img_path, result_img_path, mask_path, crop=False):
    input_img = np.array(Image.open(input_img_path).convert("RGB"))
    result_img = np.array(Image.open(result_img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    
    if crop:
        binary_mask = mask
        x, y, size = find_smallest_bounding_square(binary_mask)
        input_img = input_img[y:y+size, x:x+size]
        result_img = result_img[y:y+size, x:x+size]
        mask = mask[y:y+size, x:x+size]
    
    assert input_img.shape == result_img.shape == (mask.shape[0], mask.shape[1], 3), \
        f"Image size mismatch! input: {input_img.shape}, result: {result_img.shape}, mask: {mask.shape}"
    
    mask_fg = (mask == 255)
    mask_bg = (mask == 0)
    
    merged_img = np.zeros_like(input_img, dtype=np.uint8)
    merged_img[mask_bg] = input_img[mask_bg]
    merged_img[mask_fg] = result_img[mask_fg]
    
    return merged_img

def get_single_score(predictor, merged_img, mask, crop=False):
    if crop:
        binary_mask = mask
        x, y, size = find_smallest_bounding_square(binary_mask)
        merged_img_cropped = merged_img[y:y+size, x:x+size]
        mask_cropped = binary_mask[y:y+size, x:x+size]
        
        mask_fg = np.array(Image.fromarray(mask_cropped).resize((64,64))).reshape((1,1,64,64))//255
    else:
        merged_img_cropped = merged_img
        mask_fg = np.array(Image.fromarray(mask).resize((64,64))).reshape((1,1,64,64))//255
    
    mask_bg = 1 - mask_fg
    
    embeddings = predictor.get_aggregate_features(merged_img_cropped, [mask_fg, mask_bg])
    remove_score = cosine_similarity(embeddings[0], embeddings[1]).item()
    return remove_score

def batch_calculate_merge_remove_scores(predictor, args):
    input_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    result_files = sorted([f for f in os.listdir(args.result_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
    
    assert len(input_files) == len(result_files) == len(mask_files), \
        f"Number of files mismatch! input: {len(input_files)}, result: {len(result_files)}, mask: {len(mask_files)}"
    assert len(input_files) > 0, "No supported image files found (.png/.jpg/.jpeg/.bmp)"
    
    results = []
    print(f"Start batch merging + ReMOVE score calculation (total {len(input_files)} groups, matched by sorted order)...")
    print("-" * 120)
    
    for idx, (input_name, result_name, mask_name) in enumerate(zip(input_files, result_files, mask_files)):
        input_path = os.path.join(args.input_dir, input_name)
        result_path = os.path.join(args.result_dir, result_name)
        mask_path = os.path.join(args.mask_dir, mask_name)
        
        try:
            start_time = time.time()
            
            merged_img = merge_images(input_path, result_path, mask_path, args.crop)
            
            mask = np.array(Image.open(mask_path).convert("L"))
            if args.crop:
                x, y, size = find_smallest_bounding_square(mask)
                mask = mask[y:y+size, x:x+size]
            
            remove_score = get_single_score(predictor, merged_img, mask, args.crop)
            cost_time = time.time() - start_time
            
            results.append({
                "index": idx + 1,
                "input_name": input_name,
                "result_name": result_name,
                "mask_name": mask_name,
                "remove_score": round(remove_score, 6),
                "cost_time": round(cost_time, 3)
            })
            
            print(f"[{idx+1:3d}/{len(input_files)}] "
                  f"Input: {input_name:>20} | Result: {result_name:>20} | Mask: {mask_name:>20} | "
                  f"ReMOVE Score: {remove_score:.6f} | Time Cost: {cost_time:.3f}s")
        
        except Exception as e:
            print(f"[{idx+1:3d}/{len(input_files)}] Processing failed! Input: {input_name} | Result: {result_name} | Mask: {mask_name} | Error: {str(e)[:50]}...")
            results.append({
                "index": idx + 1,
                "input_name": input_name,
                "result_name": result_name,
                "mask_name": mask_name,
                "remove_score": "error",
                "cost_time": "error"
            })
    
    valid_scores = [r["remove_score"] for r in results if r["remove_score"] != "error"]
    print("-" * 120)
    print(f"Batch calculation completed! Total processed {len(input_files)} groups, valid samples {len(valid_scores)} groups")
    if valid_scores:
        avg_score = np.mean(valid_scores)
        print(f"Average ReMOVE Score: {avg_score:.6f}")
    else:
        print("No valid samples, cannot calculate average score")
    
    if args.save_csv:
        csv_path = args.save_csv if args.save_csv.endswith(".csv") else f"{args.save_csv}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "input_name", "result_name", "mask_name", "remove_score", "cost_time"])
            writer.writeheader()
            writer.writerows(results)
            if valid_scores:
                writer.writerow({
                    "index": "average", "input_name": "", "result_name": "", "mask_name": "",
                    "remove_score": f"{avg_score:.6f}", "cost_time": ""
                })
        print(f"Results saved to: {csv_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReMOVE Batch Test (merge: input outside mask + result inside mask)")
    parser.add_argument("-ind", "--input_dir", type=str, required=True, help="Path to original input image directory (background: outside mask)")
    parser.add_argument("-rd", "--result_dir", type=str, required=True, help="Path to result image directory (foreground: inside mask)")
    parser.add_argument("-md", "--mask_dir", type=str, required=True, help="Path to mask directory (matched with input/result by sorted order)")
    parser.add_argument("--crop", action="store_true", default=False, help="Crop images by mask bounding box (unified crop before merging)")
    parser.add_argument("--save_csv", type=str, default="merge_remove_batch_results.csv", help="CSV filename for saving results (default: merge_remove_batch_results.csv)")
    parser.add_argument("--sam_checkpoint", type=str, default="ReMOVE/models/sam_vit_h_4b8939.pth", help="Path to SAM model checkpoint (default: ReMOVE/models/sam_vit_h_4b8939.pth)")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM model type (default: vit_h)")

    args = parser.parse_args()

    if args.sam_checkpoint is None:
        args.sam_checkpoint = os.path.join(remove_dir, "models", "sam_vit_h_4b8939.pth")
    args.sam_checkpoint = os.path.abspath(args.sam_checkpoint)
    
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found at: {args.sam_checkpoint}\nPlease check the path or specify --sam_checkpoint manually.")

    print(f"Loading SAM model: {args.sam_model_type} (checkpoint path: {args.sam_checkpoint})")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).cuda()
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully!")

    batch_calculate_merge_remove_scores(predictor, args)
