import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

LOCAL_PATH = "/home/avmoura_linux/Documents/unb/sandbox_sam3"

CONFIG = {
    "images_dir": Path(f"{LOCAL_PATH}/ph2_dataset/valid/"),
    "json_file": Path(f"{LOCAL_PATH}/old_logs/segmentation/ph2_experiment_segmentation_single_gpu/dumps/ph2/coco_predictions_segm.json"),
    "output_dir": Path(f"{LOCAL_PATH}/utils/visual_results_debug_seg"),
    "score_threshold": 0.2,
    "bbox_color": (0, 255, 255),
    "valid_extensions": {'.png', '.jpg', '.jpeg', '.bmp'}
}

def load_predictions(json_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Reads the JSON file and groups predictions by image_id."""
    print(f"--> Reading JSON: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"ERROR: JSON not found at {json_path}")

    with open(json_path, 'r') as f:
        preds = json.load(f)

    if not preds:
        raise ValueError("ERROR: The JSON file is empty!")

    # Simple Statistics
    scores = [p['score'] for p in preds]
    print(f"    Total predictions: {len(preds)}")
    print(f"    Min Score: {min(scores):.5f}")
    print(f"    Max Score: {max(scores):.5f}")

    # Grouping
    preds_map = {}
    for p in preds:
        img_id = p['image_id']
        preds_map.setdefault(img_id, []).append(p)
    
    print(f"    Predictions grouped for {len(preds_map)} distinct images.")
    return preds_map

def list_images(directory: Path, extensions: set) -> List[Path]:
    """Lists and sorts images alphabetically from the directory."""
    print(f"--> Listing images in: {directory}")
    
    if not directory.exists():
        raise FileNotFoundError(f"ERROR: Image directory not found: {directory}")

    # Important: SAM3/COCO usually relies on alphabetical order for IDs if not specified otherwise
    files = sorted([
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in extensions
    ])
    
    print(f"    Total images found: {len(files)}")
    return files

def draw_annotations(img: np.ndarray, predictions: List[Dict], threshold: float) -> int:
    """Draws bounding boxes on the image and returns the count of drawn boxes."""
    count = 0
    for box in predictions:
        score = box['score']
        if score < threshold:
            continue

        x, y, w, h = map(int, box['bbox'])
        
        # Draw Rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), CONFIG["bbox_color"], 2)
        
        # Draw Text with background for better visibility
        label = f"{score:.3f}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Text Background (filled rectangle)
        cv2.rectangle(img, (x, y - 20), (x + w_text, y), CONFIG["bbox_color"], -1) 
        # Text Label (Black text on Yellow background)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        
        count += 1
    return count

def main():
    # Create output directory if it doesn't exist
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    try:
        preds_map = load_predictions(CONFIG["json_file"])
        image_files = list_images(CONFIG["images_dir"], CONFIG["valid_extensions"])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    print(f"--> Starting processing (Threshold: {CONFIG['score_threshold']})...")
    
    saved_count = 0

    # Iterate over images (assuming list index matches COCO image_id)
    for img_id, img_path in enumerate(image_files):
        if img_id not in preds_map:
            # Uncomment below to see missing IDs
            # print(f"    Warning: ID {img_id} ({img_path.name}) has no predictions.")
            continue

        # Load Image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [WARNING] Failed to read image: {img_path.name}")
            continue

        # Draw
        drawn_boxes_count = draw_annotations(img, preds_map[img_id], CONFIG["score_threshold"])

        # Save only if boxes were drawn
        if drawn_boxes_count > 0:
            output_name = f"debug_{img_path.stem}.jpg"
            output_path = CONFIG["output_dir"] / output_name
            
            cv2.imwrite(str(output_path), img)
            saved_count += 1
            
            if saved_count == 1:
                print(f"    [SUCCESS] First example saved at: {output_path}")

    print("-" * 30)
    print(f"SUMMARY: {saved_count} images saved in '{CONFIG['output_dir']}'")
    
    if saved_count == 0:
        print("DIAGNOSTIC: No images were saved. Please check:")
        print("1. If image IDs align with the JSON data.")
        print("2. If the Threshold is low enough (your scores might be < 0.02).")

if __name__ == "__main__":
    main()