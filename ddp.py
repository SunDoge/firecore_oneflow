import oneflow as flow
import oneflow.nn as nn
from icecream import ic
import oneflow.env as dist

PLACEMENT = flow.placement("cuda", [x for x in range(2)])

x = flow.zeros(2, 3).to("cuda")
ic(x)
xg = x.to_global(PLACEMENT, sbp=flow.sbp.broadcast)
ic(xg)
if dist.get_rank() == 0:
    xx = flow.ones(2, 3) * 1.0
else:
    xx = flow.ones(2, 3) * 2.0
ic(xx)
xx = xx.to_global(PLACEMENT, sbp=flow.sbp.partial_sum)
ic(xx)

xg += xx

xl = xg.to_local()
ic(xl)
