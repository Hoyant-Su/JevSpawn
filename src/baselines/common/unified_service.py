from baselines.common.history_service import HistoryBatchService
from baselines.common.streaming_service import StreamingBatchService


class UnifiedInferenceService(StreamingBatchService, HistoryBatchService):
    """Shared batched decoding with exact history reuse and optional streaming consumers."""
