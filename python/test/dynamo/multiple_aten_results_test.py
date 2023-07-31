import logging
import unittest

from shark_turbine.dynamo.importer import FxImporter
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx import (
    GraphModule,
)

class RandomTests(unittest.TestCase):
    def testInitialize(self):
        imp = FxImporter()
        print(imp.module)

    def testMultiHeadAttentionModule(self):
        imp = FxImporter()

        def import_compiler(gm: GraphModule, example_inputs):
            gm.print_readable()
            try:
                imp.import_graph_module(gm)
            finally:
                print(imp.module)
            imp.module.operation.verify()
            return gm

        backend = import_compiler
        backend = aot_autograd(fw_compiler=backend)


        import torch.nn as nn
        import torch.nn.functional as F

        class Scaled_Dot_Product_Attention(nn.Module):
            """Scaled Dot-Product Attention """

            def __init__(self):
                super(Scaled_Dot_Product_Attention, self).__init__()

            def forward(self, Q, K, V, scale=None):
                """
                Args:
                    Q: [batch_size, len_Q, dim_Q]
                    K: [batch_size, len_K, dim_K]
                    V: [batch_size, len_V, dim_V]
                    scale: 缩放因子 论文为根号dim_K
                Return:
                    self-attention后的张量，以及attention张量
                """
                attention = torch.matmul(Q, K.permute(0, 2, 1))
                if scale:
                    attention = attention * scale
                attention = F.softmax(attention, dim=-1)
                context = torch.matmul(attention, V)
                return context


        class Multi_Head_Attention(nn.Module):

            def __init__(self, dim_model, num_head, dropout=0.0):
                super(Multi_Head_Attention, self).__init__()
                self.num_head = num_head
                assert dim_model % num_head == 0
                self.dim_head = dim_model // self.num_head
                self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
                self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
                self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
                self.attention = Scaled_Dot_Product_Attention()
                self.fc = nn.Linear(num_head * self.dim_head, dim_model)
                self.dropout = nn.Dropout(dropout)
                self.layer_norm = nn.LayerNorm(dim_model)

            def forward(self, x):
                batch_size = x.size(0)
                Q = self.fc_Q(x)
                K = self.fc_K(x)
                V = self.fc_V(x)
                Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
                K = K.view(batch_size * self.num_head, -1, self.dim_head)
                V = V.view(batch_size * self.num_head, -1, self.dim_head)
                scale = K.size(-1) ** -0.5
                context = self.attention(Q, K, V, scale)
                context = context.view(batch_size, -1, self.dim_head * self.num_head)
                out = self.fc(context)
                out = self.dropout(out)
                out = out + x
                out = self.layer_norm(out)
                return out

        mod = Multi_Head_Attention(256, 4)
        opt_mod = torch.compile(mod, backend=backend)
        opt_mod(torch.randn(1, 1, 256, 256))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
