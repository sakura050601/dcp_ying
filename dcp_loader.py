import torch
from model import DCP


class DummyArgs:
    def __init__(self):
        self.emb_dims = 512
        self.n_blocks = 1
        self.dropout = 0.0
        self.ff_dims = 1024
        self.n_heads = 4
        self.cycle = False
        self.emb_nn = "dgcnn"
        self.pointer = "transformer"
        self.head = "svd"


# class DCPMatcher:
#     def __init__(self, ckpt_path=None):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # -------- 1) 构建模型结构 ----------
#         args = DummyArgs()
#         self.model = DCP(args).to(self.device)

#         # -------- 2) 读取 checkpoint ----------
#         if ckpt_path:
#             ckpt = torch.load(ckpt_path, map_location=self.device)

#             # 如果是 {"model_state_dict": xx}
#             if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#                 state_dict = ckpt["model_state_dict"]
#             else:
#                 # 直接就是 state_dict
#                 state_dict = ckpt

#             print(f"[DCPMatcher] Loaded checkpoint with {len(state_dict.keys())} keys")

#             # -------- 3) 加载参数，忽略不匹配的 ----------
#             self.model.load_state_dict(state_dict, strict=False)

#         self.model.eval()

#     def match(self, pc_src, pc_tgt):
#         src = torch.tensor(pc_src, dtype=torch.float32).unsqueeze(0).to(self.device)
#         tgt = torch.tensor(pc_tgt, dtype=torch.float32).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             R, t, _, _ = self.model(src, tgt)

#         return R.cpu().numpy()[0], t.cpu().numpy()[0]


class DCPMatcher:
    def __init__(self, ckpt_path):
        import torch
        from model import DCP

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = DummyArgs()
        self.model = DCP(args).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        print(f"[DCPMatcher] Loaded checkpoint with {len(ckpt.keys())} keys")
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

    def match(self, pc_src, pc_tgt):
        import torch

        src = torch.tensor(pc_src, dtype=torch.float32).T.unsqueeze(0).to(self.device)
        tgt = torch.tensor(pc_tgt, dtype=torch.float32).T.unsqueeze(0).to(self.device)
        with torch.no_grad():
            R, t, _, _ = self.model(src, tgt)
        return R.cpu().numpy()[0], t.cpu().numpy()[0]
