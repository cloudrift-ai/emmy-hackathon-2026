"""Problems under pinned Emmy schedules (EMMY_KNOBS). Generate with tools/generate.py."""
import numpy as np
f64 = np.float64; F = np.float32
def mm(a, b): return (a.astype(f64) @ b.astype(f64)).astype(F)
def sm(x, axis=-1):
    m = x.max(axis=axis, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=axis, keepdims=True)
def attn(q, kt, v, s): return (sm((q.astype(f64) @ kt.astype(f64)) / s) @ v.astype(f64)).astype(F)
RD = dict(rtol=1e-4, atol=1e-4); MM = dict(rtol=1e-3, atol=1e-3); FU = dict(rtol=1e-3, atol=1e-4)
OPEN = []
def add(id, code, ref, knobs, tol):
    OPEN.append(dict(id=id, code=code, ref=ref, knobs=knobs, **tol))

ROWSUM = ("torch.randn(37, 1000).sum(-1)", lambda x: [x.astype(f64).sum(-1).astype(F)])
TOTAL = ("torch.randn(70001).sum()", lambda x: [np.array(x.astype(f64).sum(), dtype=F)])
AMAX = ("torch.randn(45, 333).amax(-1)", lambda x: [x.max(-1)])
COLSUM = ("torch.randn(999, 40).sum(0)", lambda x: [x.astype(f64).sum(0).astype(F)])
for w in ("t32", "t64", "t256", "t512", "t1024"):
    add(f"kr_rowsum_coop_{w}", *ROWSUM, f"WORK={w},REDUCE=coop", RD)
    add(f"kr_total_coop_{w}", *TOTAL, f"WORK={w},REDUCE=coop", dict(rtol=1e-3, atol=1e-3))
for r in ("coop-t", "g4", "g8", "g4a", "g4k", "r4", "r2", ""):
    add(f"kr_rowsum_{r or 'serial'}", *ROWSUM, f"WORK=t128,REDUCE={r}", RD)
    add(f"kr_colsum_{r or 'serial'}", *COLSUM, f"WORK=t128,REDUCE={r}", RD)
    add(f"kr_amax_{r or 'serial'}", *AMAX, f"WORK=t128,REDUCE={r}", dict(rtol=0, atol=0))
SOFT = ("torch.softmax(torch.randn(29, 515), dim=-1)", lambda x: [sm(x.astype(f64)).astype(F)])
LN = ("torch.nn.functional.layer_norm(torch.randn(19, 300), (300,))",
      lambda x: [((x.astype(f64) - x.astype(f64).mean(-1, keepdims=True)) / np.sqrt(x.astype(f64).var(-1, keepdims=True) + 1e-5)).astype(F)])
for k in ("WORK=t32,REDUCE=coop", "WORK=t256,REDUCE=coop", "WORK=t1024,REDUCE=coop", "WORK=t128,REDUCE=coop-t", "WORK=t128,REDUCE=r4", "WORK=t64,REDUCE="):
    tag = k.replace("WORK=", "").replace(",REDUCE=", "_").replace("-", "") or "x"
    add(f"kf_softmax_{tag}", *SOFT, k, dict(rtol=1e-3, atol=1e-5))
    add(f"kf_layernorm_{tag}", *LN, k, FU)
MMX = ("torch.randn(200, 136) @ torch.randn(136, 168)", lambda a, b: [mm(a, b)])
for k in ("TILE=f1", "TILE=f2x2", "TILE=f4x4", "TILE=f8x2", "TILE=f4x4,STAGE=d2/smem", "TILE=f4x4,STAGE=d1/smem",
          "TILE=f2x2,STAGE=d2/smem-async", "TILE=f4x4,STAGE=d3/smem-async", "TILE=f4x4,STAGE=d4/smem-async",
          "TILE=f4x4,STAGE=d2/smem-async/p2", "TILE=f4x4,STAGE=d2/smem-async,RASTER=gm3", "TILE=f4x4,STAGE=d2/smem-async,RASTER=gn5",
          "TILE=f2x2,REDUCE=g4", "TILE=f2x2,REDUCE=g4k", "TILE=f4x4,WORK=t64", "TILE=f4x4,WORK=t512"):
    tag = k.replace("TILE=", "").replace(",STAGE=", "_").replace(",RASTER=", "_").replace(",REDUCE=", "_").replace(",WORK=", "_").replace("/", "").replace("-", "")
    add(f"km_{tag}", *MMX, k, MM)
ATT = ("q = torch.randn(72, 40); kt = torch.randn(40, 90); v = torch.randn(90, 24); torch.softmax(q @ kt / 6.324555, dim=-1) @ v",
       lambda q, kt, v: [attn(q, kt, v, 6.324555)])
for k in ("", "TILE=f2x2", "TILE=f4x4", "TILE=f1", "TILE=f2x2,STAGE=d2/smem-async", "TILE=f2x2,RASTER=gm3", "WORK=t128,REDUCE=coop"):
    tag = (k.replace("TILE=", "").replace(",STAGE=", "_").replace(",RASTER=", "_").replace("WORK=", "").replace(",REDUCE=", "_").replace("/", "").replace("-", "")) or "default"
    add(f"ka_{tag}", *ATT, k, FU)

for shape, (m, k, n) in (("sq512", (512, 512, 512)), ("r384", (384, 256, 320)), ("t256", (256, 512, 128))):
    code = f"torch.randn({m}, {k}) @ torch.randn({k}, {n})"
    for kn in ("", "STAGE=d2/smem", "STAGE=d1/smem", "STAGE=d2/smem-async", "STAGE=d3/smem-async", "STAGE=d4/smem-async",
               "STAGE=d2/smem-async/p2", "STAGE=d3/smem-async,RASTER=gm3", "STAGE=d3/smem-async,RASTER=gn5",
               "STAGE=d3/smem-async,RASTER=gm8", "STAGE=d2/smem,RASTER=gm3", "RASTER=gm3"):
        tag = (kn.replace("STAGE=", "").replace(",RASTER=", "_").replace("RASTER=", "").replace("/", "").replace("-", "")) or "default"
        add(f"ks_{shape}_{tag}", code, lambda a, b: [mm(a, b)], kn, MM)
for nm, code, ref in (("rowsum", "torch.randn(37, 1024).sum(-1)", lambda x: [x.astype(f64).sum(-1).astype(F)]),
                      ("colsum", "torch.randn(1024, 40).sum(0)", lambda x: [x.astype(f64).sum(0).astype(F)]),
                      ("amax", "torch.randn(45, 512).amax(-1)", lambda x: [x.max(-1)])):
    for r in ("g4k", "g8k", "g4a", "g8a", "g2k"):
        add(f"kg_{nm}_{r}", code, ref, f"WORK=t128,REDUCE={r}", RD if nm != "amax" else dict(rtol=0, atol=0))
