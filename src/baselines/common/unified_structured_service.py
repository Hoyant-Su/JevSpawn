from baselines.common.scored_finite_service import ScoredFiniteService
from baselines.common.unified_service import UnifiedInferenceService


class UnifiedStructuredService(ScoredFiniteService, UnifiedInferenceService):
    """Finite native readout and ordinary generation share the same text inference service."""
