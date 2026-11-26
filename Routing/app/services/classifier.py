from app.core.config import Settings
from app.prompts import CLASSIFIER_PROMPT
from app.providers.base import JSONProvider
from app.schemas import Classification


def classify(provider: JSONProvider, settings: Settings, message: str) -> Classification:
    """Label a ticket with one route. This only classifies; branching happens in the router."""
    return provider.complete_json(
        system_prompt=CLASSIFIER_PROMPT,
        payload={"ticket": message},
        response_model=Classification,
        model=settings.model_for("classifier"),
    )
