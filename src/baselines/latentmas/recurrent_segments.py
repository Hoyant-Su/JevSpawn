from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5

from baselines.latentmas.native_hybrid_transport import ChunkedHybridTransport


class RecurrentSegmentTransport(ChunkedHybridTransport):
    def _forward(self, embeddings, mask, cache):
        original = modeling_qwen3_5.torch_chunk_gated_delta_rule
        # Keep the decode kernel's FP32 normalization and recurrent update order.
        # The owning inference service serializes model forwards in this process.
        modeling_qwen3_5.torch_chunk_gated_delta_rule = fused_recurrent_gated_delta_rule
        try:
            return super()._forward(embeddings, mask, cache)
        finally:
            modeling_qwen3_5.torch_chunk_gated_delta_rule = original
