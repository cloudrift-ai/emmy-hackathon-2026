"""Edge-case problems in the five categories. Generate with tools/generate.py."""

import math

import numpy as np

f64 = np.float64
F = np.float32
R = "WORK=t128,REDUCE=coop"


def _erf(x):
    return np.vectorize(math.erf)(x)


def _sm(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def mm(a, b):
    return (a.astype(f64) @ b.astype(f64)).astype(F)


def attn(q, kt, v, scale, mask=None):
    s = (q.astype(f64) @ kt.astype(f64)) / scale
    if mask is not None:
        s = np.where(mask, -np.inf, s)
    return (_sm(s) @ v.astype(f64)).astype(F)


def d(x):
    return x.astype(f64)


def c(x):
    return [np.asarray(x).astype(F)]


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


P = dict(rtol=1e-5, atol=1e-6)
EX = dict(rtol=0, atol=0)
RD = dict(rtol=1e-4, atol=1e-4)
MM = dict(rtol=1e-3, atol=1e-3)
FU = dict(rtol=1e-3, atol=1e-4)

OPEN = [
    # ---------------- pointwise: sizes ----------------
    dict(id="ap01_size1", code="torch.randn(1) + torch.randn(1)", ref=lambda a, b: [a + b], **EX),
    dict(id="ap02_size3", code="torch.randn(3) * torch.randn(3)", ref=lambda a, b: [a * b], **EX),
    dict(id="ap03_size31", code="torch.relu(torch.randn(31))", ref=lambda a: [np.maximum(a, 0)], **EX),
    dict(id="ap04_size33", code="torch.randn(33) - torch.randn(33)", ref=lambda a, b: [a - b], **EX),
    dict(id="ap05_size257", code="torch.randn(257) * 2.0 + 1.0", ref=lambda a: [a * F(2) + F(1)], **EX),
    dict(id="ap06_prime", code="torch.randn(100003) + torch.randn(100003)", ref=lambda a, b: [a + b], **EX),
    dict(id="ap07_2d_odd", code="torch.randn(7, 13) * torch.randn(7, 13)", ref=lambda a, b: [a * b], **EX),
    dict(id="ap08_3d_bcast", code="torch.randn(4, 1, 5) + torch.randn(1, 3, 1)", ref=lambda a, b: [a + b], **EX),
    dict(id="ap09_bcast_scalar_t", code="torch.randn(37, 61) * torch.randn(1, 1)", ref=lambda a, b: [a * b], **EX),
    dict(id="ap10_div_tensor", code="torch.randn(4096) / torch.randn(4096)", ref=lambda a, b: [a / b], **EX),
    dict(id="ap11_div_const", code="torch.randn(4096) / 3.0", ref=lambda a: c(d(a) / 3.0), **P),
    # ---------------- pointwise: ops / magnitudes ----------------
    dict(id="ap12_exp_big", code="torch.exp(torch.randn(4096) * 30.0)", ref=lambda a: c(np.exp(d(a * F(30)))), **P),
    dict(id="ap13_sigmoid_big", code="torch.sigmoid(torch.randn(4096) * 50.0)", ref=lambda a: c(1 / (1 + np.exp(-d(a * F(50))))), **P),
    dict(id="ap14_tanh_big", code="torch.tanh(torch.randn(4096) * 20.0)", ref=lambda a: c(np.tanh(d(a * F(20)))), **P),
    dict(id="ap15_tanh_small", code="torch.tanh(torch.randn(4096) * 0.001)", ref=lambda a: c(np.tanh(d(a * F(0.001)))), **P),
    dict(id="ap16_gelu_tanh", code="torch.nn.functional.gelu(torch.randn(4096), approximate='tanh')",
         ref=lambda x: c(0.5 * d(x) * (1 + np.tanh(np.sqrt(2 / np.pi) * (d(x) + 0.044715 * d(x) ** 3)))), **P),
    dict(id="ap17_softplus_big", code="torch.nn.functional.softplus(torch.randn(4096) * 30.0)", ref=lambda a: c(softplus(d(a * F(30)))), **P),
    dict(id="ap18_silu_big", code="torch.nn.functional.silu(torch.randn(4096) * 40.0)",
         ref=lambda a: c(d(a * F(40)) / (1 + np.exp(-d(a * F(40))))), **P),
    dict(id="ap19_pow_frac", code="x = torch.randn(4096); torch.abs(x) ** 2.5", ref=lambda x: c(np.abs(d(x)) ** 2.5), **P),
    dict(id="ap20_pow_neg", code="x = torch.randn(4096); (torch.abs(x) + 0.5) ** -1.5", ref=lambda x: c((np.abs(d(x)) + 0.5) ** -1.5), **P),
    dict(id="ap21_pow_int7", code="torch.randn(4096) ** 7", ref=lambda x: c(d(x) ** 7), **P),
    dict(id="ap22_sqrt_abs", code="torch.sqrt(torch.abs(torch.randn(4096)))", ref=lambda x: c(np.sqrt(np.abs(d(x)))), **P),
    dict(id="ap23_log_tiny", code="torch.log(torch.abs(torch.randn(4096)) * 1e-30)", ref=lambda x: c(np.log(d(np.abs(x) * F(1e-30)))), **P),
    dict(id="ap24_erf", code="torch.erf(torch.randn(4096) * 3.0)", ref=lambda x: c(_erf(d(x * F(3)))), **P),
    dict(id="ap25_sin_big", code="torch.sin(torch.randn(4096) * 100.0)", ref=lambda x: c(np.sin(d(x * F(100)))), **P),
    dict(id="ap26_cos", code="torch.cos(torch.randn(4096) * 10.0)", ref=lambda x: c(np.cos(d(x * F(10)))), **P),
    dict(id="ap27_where_eq", code="x = torch.randn(2048); torch.where(x >= 0.5, x * 2.0, -x)", ref=lambda x: [np.where(x >= F(0.5), x * F(2), -x)], **EX),
    dict(id="ap28_mask_float", code="x = torch.randn(2048); (x > 0).float() * x", ref=lambda x: [(x > 0).astype(F) * x], **EX),
    dict(id="ap29_mask_and", code="x = torch.randn(2048); y = torch.randn(2048); torch.where((x > 0) & (y < 0), x, y)",
         ref=lambda x, y: [np.where((x > 0) & (y < 0), x, y)], **EX),
    dict(id="ap30_clamp_one_side", code="torch.clamp(torch.randn(3000), max=0.25)", ref=lambda x: [np.minimum(x, F(0.25))], **EX),
    dict(id="ap31_tril", code="torch.tril(torch.randn(33, 33))", ref=lambda x: [np.tril(x)], **EX),
    dict(id="ap32_chain", code="x = torch.randn(1000); torch.sigmoid(x) * torch.tanh(x) + torch.relu(-x) / (torch.abs(x) + 1.0)",
         ref=lambda x: c(1 / (1 + np.exp(-d(x))) * np.tanh(d(x)) + np.maximum(-d(x), 0) / (np.abs(d(x)) + 1)), **P),
    # ---------------- reductions ----------------
    dict(id="ar01_sum_1", code="torch.randn(1).sum()", ref=lambda x: c(d(x).sum()), knobs=R, **RD),
    dict(id="ar02_sum_7", code="torch.randn(7).sum()", ref=lambda x: c(d(x).sum()), knobs=R, **RD),
    dict(id="ar03_sum_129", code="torch.randn(129).sum()", ref=lambda x: c(d(x).sum()), knobs=R, **RD),
    dict(id="ar04_amax_4097", code="torch.randn(4097).amax()", ref=lambda x: [x.max(keepdims=False)], knobs=R, **EX),
    dict(id="ar05_amin_rows", code="torch.randn(33, 77).amin(-1)", ref=lambda x: [x.min(-1)], knobs=R, **EX),
    dict(id="ar06_rows_short", code="torch.randn(1000, 3).sum(-1)", ref=lambda x: c(d(x).sum(-1)), knobs=R, **RD),
    dict(id="ar07_rows_one", code="torch.randn(5, 1).sum(-1)", ref=lambda x: c(d(x).sum(-1)), knobs=R, **RD),
    dict(id="ar08_rows_long", code="torch.randn(3, 20000).sum(-1)", ref=lambda x: c(d(x).sum(-1)), knobs=R, **RD),
    dict(id="ar09_mid_axis", code="torch.randn(6, 50, 7).sum(1)", ref=lambda x: c(d(x).sum(1)), knobs=R, **RD),
    dict(id="ar10_first_axis_odd", code="torch.randn(301, 5).mean(0)", ref=lambda x: c(d(x).mean(0)), knobs=R, **RD),
    dict(id="ar11_keepdim", code="torch.randn(16, 100).amax(-1, keepdim=True)", ref=lambda x: [x.max(-1, keepdims=True)], knobs=R, **EX),
    dict(id="ar12_multi_axis", code="torch.randn(8, 9, 10).sum((0, 2))", ref=lambda x: c(d(x).sum((0, 2))), knobs=R, **RD),
    dict(id="ar13_unpinned_big", code="torch.randn(3000001).sum()", ref=lambda x: c(d(x).sum()), rtol=1e-3, atol=2e-2),
    dict(id="ar14_unpinned_rows_odd", code="torch.randn(77, 3001).sum(-1)", ref=lambda x: c(d(x).sum(-1)), rtol=1e-4, atol=1e-3),
    dict(id="ar15_unpinned_amax", code="torch.randn(500000).amax()", ref=lambda x: [x.max()], **EX),
    dict(id="ar16_sumsq_mean", code="x = torch.randn(4099); (x * x).mean()", ref=lambda x: c((d(x) ** 2).mean()), knobs=R, **RD),
    dict(id="ar17_neg_amax", code="(-torch.abs(torch.randn(2048))).amax()", ref=lambda x: [(-np.abs(x)).max()], knobs=R, **EX),
    # ---------------- matmul ----------------
    dict(id="am01_1xK_KxN", code="torch.randn(1, 64) @ torch.randn(64, 48)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am02_outer", code="torch.randn(40, 1) @ torch.randn(1, 30)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am03_tiny", code="torch.randn(3, 4) @ torch.randn(4, 5)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am04_odd", code="torch.randn(17, 33) @ torch.randn(33, 9)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am05_odd_big", code="torch.randn(129, 257) @ torch.randn(257, 65)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am06_N1", code="torch.randn(96, 80) @ torch.randn(80, 1)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am07_huge_k", code="torch.randn(16, 8192) @ torch.randn(8192, 16)", ref=lambda a, b: [mm(a, b)], rtol=1e-3, atol=3e-3),
    dict(id="am08_wide_n", code="torch.randn(8, 64) @ torch.randn(64, 2000)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am09_batched_odd", code="torch.randn(3, 17, 31) @ torch.randn(3, 31, 13)", ref=lambda a, b: [mm(a, b)], **MM),
    dict(id="am10_mm_gelu", code="a = torch.randn(64, 96); b = torch.randn(96, 80); torch.nn.functional.gelu(a @ b)",
         ref=lambda a, b: c(0.5 * d(mm(a, b)) * (1 + _erf(d(mm(a, b)) / np.sqrt(2)))), rtol=1e-3, atol=1e-3),
    dict(id="am11_linear_bias", code="a = torch.randn(50, 70); w = torch.randn(70, 30); b = torch.randn(30); a @ w + b",
         ref=lambda a, w, b: c(d(mm(a, w)) + b), **MM),
    dict(id="am12_square_1024", code="torch.randn(1024, 1024) @ torch.randn(1024, 1024)", ref=lambda a, b: [mm(a, b)], rtol=1e-3, atol=2e-3),
    # ---------------- fused ----------------
    dict(id="af01_softmax_len1", code="torch.softmax(torch.randn(64, 1), dim=-1)", ref=lambda x: c(_sm(d(x))), rtol=1e-3, atol=1e-5),
    dict(id="af02_softmax_len3", code="torch.softmax(torch.randn(300, 3), dim=-1)", ref=lambda x: c(_sm(d(x))), rtol=1e-3, atol=1e-5),
    dict(id="af03_softmax_len33", code="torch.softmax(torch.randn(31, 33), dim=-1)", ref=lambda x: c(_sm(d(x))), rtol=1e-3, atol=1e-5),
    dict(id="af04_softmax_2049", code="torch.softmax(torch.randn(5, 2049), dim=-1)", ref=lambda x: c(_sm(d(x))), rtol=1e-3, atol=1e-5),
    dict(id="af05_softmax_hot", code="torch.softmax(torch.randn(16, 128) * 100.0, dim=-1)", ref=lambda x: c(_sm(d(x * F(100)))), rtol=1e-3, atol=1e-5),
    dict(id="af06_softmax_dim0_odd", code="torch.softmax(torch.randn(77, 9), dim=0)", ref=lambda x: c(_sm(d(x), 0)), rtol=1e-3, atol=1e-5),
    dict(id="af07_layernorm_odd", code="torch.nn.functional.layer_norm(torch.randn(13, 777), (777,))",
         ref=lambda x: c((d(x) - d(x).mean(-1, keepdims=True)) / np.sqrt(d(x).var(-1, keepdims=True) + 1e-5)), **FU),
    dict(id="af08_layernorm_affine", code="x = torch.randn(64, 768); w = torch.randn(768); b = torch.randn(768); torch.nn.functional.layer_norm(x, (768,), w, b)",
         ref=lambda x, w, b: c((d(x) - d(x).mean(-1, keepdims=True)) / np.sqrt(d(x).var(-1, keepdims=True) + 1e-5) * w + b), **FU),
    dict(id="af09_rmsnorm_100", code="x = torch.randn(10, 100); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)",
         ref=lambda x: c(d(x) / np.sqrt((d(x) ** 2).mean(-1, keepdims=True) + 1e-6)), **FU),
    dict(id="af10_log_softmax", code="torch.log_softmax(torch.randn(40, 300), dim=-1)",
         ref=lambda x: c(d(x) - d(x).max(-1, keepdims=True) - np.log(np.exp(d(x) - d(x).max(-1, keepdims=True)).sum(-1, keepdims=True))), **FU),
    # torch.ones is traced as a seeded input x
    dict(id="af11_masked_softmax", code="s = torch.randn(32, 32); torch.softmax(s.masked_fill(torch.triu(torch.ones(32, 32), diagonal=1) > 0, float('-inf')), dim=-1)",
         ref=lambda s, x: c(_sm(np.where(np.triu(x, 1) > 0, -np.inf, d(s)))), rtol=1e-3, atol=1e-5),
    dict(id="af12_center_scale", code="x = torch.randn(50, 257); (x - x.mean(-1, keepdim=True)) * torch.sigmoid(x.amax(-1, keepdim=True))",
         ref=lambda x: c((d(x) - d(x).mean(-1, keepdims=True)) / (1 + np.exp(-d(x).max(-1, keepdims=True)))), **FU),
    # ---------------- attention ----------------
    dict(id="aa01_one_query", code="q = torch.randn(1, 32); kt = torch.randn(32, 64); v = torch.randn(64, 32); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [attn(q, kt, v, 5.656854)], **FU),
    dict(id="aa02_one_key", code="q = torch.randn(16, 8); kt = torch.randn(8, 1); v = torch.randn(1, 8); torch.softmax(q @ kt / 2.828427, dim=-1) @ v",
         ref=lambda q, kt, v: [attn(q, kt, v, 2.828427)], **FU),
    dict(id="aa03_odd_seq", code="q = torch.randn(17, 24); kt = torch.randn(24, 33); v = torch.randn(33, 40); torch.softmax(q @ kt / 4.898979, dim=-1) @ v",
         ref=lambda q, kt, v: [attn(q, kt, v, 4.898979)], **FU),
    dict(id="aa04_head128", code="q = torch.randn(64, 128); kt = torch.randn(128, 64); v = torch.randn(64, 128); torch.softmax(q @ kt / 11.313708, dim=-1) @ v",
         ref=lambda q, kt, v: [attn(q, kt, v, 11.313708)], **FU),
    dict(id="aa05_long_kv", code="q = torch.randn(32, 64); kt = torch.randn(64, 1000); v = torch.randn(1000, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [attn(q, kt, v, 8.0)], **FU),
    # q @ kt / scale is traced as one input s, torch.ones as a seeded input x
    dict(id="aa06_causal", code="q = torch.randn(48, 32); kt = torch.randn(32, 48); v = torch.randn(48, 32); s = q @ kt / 5.656854; "
                                "torch.softmax(s.masked_fill(torch.triu(torch.ones(48, 48), diagonal=1) > 0, float('-inf')), dim=-1) @ v",
         ref=lambda s, x, v: c(_sm(np.where(np.triu(x, 1) > 0, -np.inf, d(s))) @ d(v)), **FU),
    dict(id="aa07_batched_odd", code="q = torch.randn(3, 20, 16); kt = torch.randn(3, 16, 37); v = torch.randn(3, 37, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [np.stack([attn(q[i], kt[i], v[i], 4.0) for i in range(3)])], **FU),
    dict(id="aa08_hot_scores", code="q = torch.randn(32, 32); kt = torch.randn(32, 32); v = torch.randn(32, 32); torch.softmax(q @ kt * 10.0, dim=-1) @ v",
         ref=lambda q, kt, v: [attn(q, kt, v, 0.1)], **FU),
]
