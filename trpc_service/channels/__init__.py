"""Provider adapters that normalize callbacks and safely deliver reply envelopes."""

from .adapters import (
    Attachment,
    CallbackRequest,
    ChannelAdapter,
    ChannelBinding,
    ChannelError,
    DeliveryCapability,
    DeliveryResult,
    InboundEnvelope,
    MockChannelAdapter,
    Principal,
    ReplyBlock,
    ReplyEnvelope,
    TelegramAdapter,
    WeComAdapter,
    deterministic_session_id,
)

__all__ = [
    "Attachment",
    "CallbackRequest",
    "ChannelAdapter",
    "ChannelBinding",
    "ChannelError",
    "DeliveryCapability",
    "DeliveryResult",
    "InboundEnvelope",
    "MockChannelAdapter",
    "Principal",
    "ReplyBlock",
    "ReplyEnvelope",
    "TelegramAdapter",
    "WeComAdapter",
    "deterministic_session_id",
]
