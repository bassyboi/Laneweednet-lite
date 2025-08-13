"""
Convert COCO boxes to 4-lane labels.

Input COCO (standard):
- "images": [{"id": int, "file_name": str, "width": int, "height": int}, ...]
- "annotations": [{"image_id": int, "bbox": [x,y,w,h], "iscrowd":0}, ...]

Output (lane-only JSON):
{
  "images": [{"file_name": "000123.jpg"}, ...],
  "lane_labels": { "000123.jpg": [1,0,1,0], ... }
}

Usage:
python scripts/lane_labels_from_coco.py --coco path/train.json --img_dir path/images --img_size 384 --lanes 4 --area_thresh 4800 --out lanes_train.json
"""
import json, argparse
from pathlib import Path

def boxes_to_lanes(boxes, img_w, lanes=4, area_thresh=4800.0):
    lane_w = img_w / lanes
    accum = [0.0]*lanes
    for (x1,y1,x2,y2) in boxes:
        x1 = max(0.0, min(img_w-1.0, x1)); x2 = max(0.0, min(img_w-1.0, x2))
        if x2 <= x1: 
            continue
        left_lane  = int(x1 // lane_w)
        right_lane = int((x2-1) // lane_w)
        bw = (x2-x1); bh = max(0.0, y2-y1); area = bw*bh
        for li in range(max(0,left_lane), min(lanes-1,right_lane)+1):
            lx1 = li*lane_w; lx2 = (li+1)*lane_w
            overlap = max(0.0, min(x2,lx2) - max(x1,lx1))
            if overlap > 0:
                accum[li] += area * (overlap / (bw + 1e-6))
    return [1 if a >= area_thresh else 0 for a in accum]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--img_dir", required=True)   # not used directly here, kept for compatibility
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--area_thresh", type=float, default=4800.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    by_img = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 1: 
            continue
        x,y,w,h = a["bbox"]
        by_img[a["image_id"]].append([x, y, x+w, y+h])

    lane_labels = {}
    for im in coco["images"]:
        W, H = im.get("width", args.img_size), im.get("height", args.img_size)
        scale = min(args.img_size / max(1.0, W), args.img_size / max(1.0, H))
        boxes = [[x*scale, y*scale, X*scale, Y*scale] for (x,y,X,Y) in by_img.get(im["id"], [])]
        lane_labels[im["file_name"]] = boxes_to_lanes(boxes, img_w=args.img_size, lanes=args.lanes, area_thresh=args.area_thresh)

    out = {"images": [{"file_name": im["file_name"]} for im in coco["images"]],
           "lane_labels": lane_labels}
    Path(args.out).write_text(json.dumps(out))
    print("Wrote", args.out)

if __name__ == "__main__":
    main()