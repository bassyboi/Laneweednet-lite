import argparse, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from rwnet.lane_model import LaneNet

def letterbox(im, S):
    W,H = im.size; sc=min(S/W,S/H); nw,nh=int(W*sc),int(H*sc)
    canvas = Image.new("RGB",(S,S),(114,114,114))
    canvas.paste(im.resize((nw,nh), Image.BILINEAR), ((S-nw)//2,(S-nh)//2))
    arr = np.array(canvas).transpose(2,0,1)
    return torch.from_numpy(arr).float()/255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs/best.pt")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--viz", type=str, default=None, help="optional: save a lane visualization")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt["cfg"]
    model = LaneNet(base=cfg.get("base",32), depth=cfg.get("depth",2), lanes=4)
    model.load_state_dict(ckpt["model"]); model.eval()

    im = Image.open(args.image).convert("RGB")
    x = letterbox(im, args.img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].tolist()
    print("lane_probs:", [round(p,3) for p in probs])

    if args.viz:
        W,H = im.size
        lanes = 4; lane_w = W/lanes
        vis = im.copy(); dr = ImageDraw.Draw(vis)
        for i,p in enumerate(probs):
            x1 = i*lane_w; x2 = (i+1)*lane_w
            col = (0, int(255*min(1.0,p)), 0)
            dr.rectangle([x1, 0, x2, H], outline=col, width=3)
            dr.text((x1+5,5), f"{p:.2f}", fill=col)
        vis.save(args.viz); print("Saved", args.viz)

if __name__ == "__main__":
    main()