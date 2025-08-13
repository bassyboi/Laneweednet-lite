import torch, torch.nn as nn, torch.nn.functional as F

def autopad(k): return k//2

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1):
        super().__init__()
        self.cv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.cv(x)))

class DWBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = Conv(c, c, 3, 1, g=c)
        self.pw = Conv(c, c, 1, 1)
    def forward(self, x): return self.pw(self.dw(x))

class LaneNet(nn.Module):
    """Tiny depthwise CNN + 4-lane head (shared FC)."""
    def __init__(self, base=32, depth=2, lanes=4):
        super().__init__()
        self.stem = Conv(3, base, 3, 2)                     # s=2
        self.s2   = nn.Sequential(*[DWBlock(base) for _ in range(depth)])
        self.d4   = Conv(base, base*2, 3, 2)                # s=4
        self.s4   = nn.Sequential(*[DWBlock(base*2) for _ in range(depth)])
        self.d8   = Conv(base*2, base*4, 3, 2)              # s=8
        self.s8   = nn.Sequential(*[DWBlock(base*4) for _ in range(depth)])
        self.head = nn.Sequential(Conv(base*4, base*4), Conv(base*4, base*4))
        self.fc   = nn.Linear(base*4, 1)   # shared per-lane
        self.lanes= lanes
    def forward(self, x):
        x = self.stem(x); x = self.s2(x)
        x = self.d4(x);  x = self.s4(x)
        x = self.d8(x);  p8 = self.s8(x)              # (B,C,H,W)
        h = self.head(p8).mean(2)                     # avg over height -> (B,C,W)
        B, C, W = h.shape; chunk = max(1, W // self.lanes)
        logits = []
        for i in range(self.lanes):
            xi = h[:, :, i*chunk:(i+1)*chunk].mean(2) # (B,C)
            logits.append(self.fc(xi))                # (B,1)
        return torch.cat(logits, dim=1)               # (B, lanes)