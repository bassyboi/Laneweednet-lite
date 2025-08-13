import argparse, json, random, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from rwnet.lane_model import LaneNet

class LaneSet(Dataset):
    def __init__(self, lanes_json, img_dir, img_size=384, augment=True):
        j = json.loads(Path(lanes_json).read_text())
        self.items = [im["file_name"] for im in j["images"]]
        self.labels = j["lane_labels"]
        self.img_dir = Path(img_dir)
        self.S = img_size; self.augment = augment
    def __len__(self): return len(self.items)
    def _augment(self, im):
        if random.random()<0.5: im = ImageOps.mirror(im)
        if random.random()<0.7: im = ImageEnhance.Color(im).enhance(random.uniform(0.7,1.3))
        if random.random()<0.7: im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7,1.3))
        if random.random()<0.7: im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8,1.2))
        return im
    def _letterbox(self, im):
        W,H = im.size; S=self.S
        sc = min(S/W, S/H); nw,nh = int(W*sc), int(H*sc)
        canvas = Image.new("RGB",(S,S),(114,114,114))
        canvas.paste(im.resize((nw,nh), Image.BILINEAR), ((S-nw)//2,(S-nh)//2))
        return canvas
    def __getitem__(self, i):
        fn = self.items[i]
        im = Image.open(self.img_dir/fn).convert("RGB")
        if self.augment: im = self._augment(im)
        im = self._letterbox(im)
        x = torch.from_numpy(np.array(im).transpose(2,0,1)).float()/255.0
        y = torch.tensor(self.labels[fn], dtype=torch.float32)
        return x, y

def train(cfg):
    dev = torch.device("cuda" if (cfg.device != "cpu" and torch.cuda.is_available()) else "cpu")
    tr = LaneSet(cfg.train, cfg.img_dir, cfg.img, augment=True)
    va = LaneSet(cfg.val,   cfg.img_dir, cfg.img, augment=False)
    tl = DataLoader(tr, batch_size=cfg.batch, shuffle=True,  num_workers=2)
    vl = DataLoader(va, batch_size=cfg.batch, shuffle=False, num_workers=2)

    model = LaneNet(base=cfg.base, depth=cfg.depth, lanes=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    posw = torch.tensor([cfg.pos_weight]*4, device=dev)
    for ep in range(1, cfg.epochs+1):
        model.train(); pb=tqdm(tl, ncols=100, desc=f"train {ep}/{cfg['epochs']}")
        for x,y in pb:
            x,y = x.to(dev), y.to(dev)
            logit = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=posw)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            pb.set_postfix(loss=f"{loss.item():.4f}")
        # quick val proxy
        model.eval(); s=0.0;c=0
        with torch.no_grad():
            for x,y in vl:
                x=x.to(dev); logit=model(x); s += torch.sigmoid(logit).mean().item(); c+=1
        print(f"val_proxy={s/max(1,c):.4f}")
        Path("runs").mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, "runs/last.pt")
        torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, "runs/best.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--pos_weight", type=float, default=1.2)
    ap.add_argument("--device", type=str, default="auto")  # auto/cpu/cuda
    args = ap.parse_args()
    train(args)