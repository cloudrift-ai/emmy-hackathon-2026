# Emmy accumulates float64 and int64 sums in f32, hence the d03/d04 tolerances
import numpy as np
f64=np.float64
OPEN=[
 dict(id="d01_f64_add", code="torch.randn(1000, dtype=torch.float64) + torch.randn(1000, dtype=torch.float64)", ref=lambda a,b:[a+b], rtol=0, atol=0),
 dict(id="d02_f64_mul_exp", code="x = torch.randn(1000, dtype=torch.float64); x * x + 1.0", ref=lambda x:[x*x+1.0], rtol=0, atol=0),
 dict(id="d03_f64_sum", code="torch.randn(5000, dtype=torch.float64).sum()", ref=lambda x:[np.array(x.sum())], rtol=1e-5, atol=1e-5, knobs="WORK=t128,REDUCE=coop"),
 dict(id="d04_i64_sum", code="torch.arange(1000).sum()", ref=lambda x:[np.array(x.sum())], rtol=0, atol=0, knobs="WORK=t128,REDUCE=coop"),
 dict(id="d05_f16_add", code="torch.randn(1000, dtype=torch.float16) + torch.randn(1000, dtype=torch.float16)", ref=lambda a,b:[a+b], rtol=1e-3, atol=1e-3),
]
