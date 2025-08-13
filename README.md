# LaneWeedNet-Lite 🌱 (4-lane, box-free)

Ultra-light model for weed spraying that predicts **4 lane probabilities** per frame (left→right).  
No bounding boxes, no NMS — just 4 logits → spray/no-spray per lane.

## Install
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Convert your COCO boxes → lane labels (one-off)
```bash
python scripts/lane_labels_from_coco.py \
  --coco /path/to/your_train.json \
  --img_dir /path/to/images \
  --img_size 384 \
  --lanes 4 \
  --area_thresh 4800 \
  --out lanes_train.json

python scripts/lane_labels_from_coco.py \
  --coco /path/to/your_val.json \
  --img_dir /path/to/images \
  --img_size 384 \
  --lanes 4 \
  --area_thresh 4800 \
  --out lanes_val.json
```

> Tip: if you don't have COCO files, you can author lane JSONs by hand; format is in the script header.

## 2) Train
```bash
python train_lanes.py --train lanes_train.json --val lanes_val.json --img_dir /path/to/images --img 384 --epochs 20
```

## 3) Inference
```bash
python infer_lanes.py --weights runs/best.pt --image /path/to/frame.jpg --img 384
# prints: lane_probs: [p0,p1,p2,p3]
```

## Notes
- Tune `--area_thresh` if conversion yields too many 1s or 0s per lane.
- Use hysteresis + EMA on the 4 probabilities on-rig to avoid flicker.
- Map camera FOV → boom geometry so each lane corresponds to a nozzle group.

MIT licensed.