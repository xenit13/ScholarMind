from __future__ import annotations

from dataclasses import dataclass

from scholar_mind.config.settings import Settings


@dataclass(slots=True)
class ChatProviderConfig:
    model: str
    base_url: str | None
    api_key: str | None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key)


@dataclass(slots=True)
class ProviderBundle:
    reasoning: ChatProviderConfig
    light: ChatProviderConfig


def build_provider_bundle(settings: Settings) -> ProviderBundle:
    return ProviderBundle(
        reasoning=ChatProviderConfig(
            model=settings.llm_reasoning_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
        light=ChatProviderConfig(
            model=settings.llm_light_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
    )
