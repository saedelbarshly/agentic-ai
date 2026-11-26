class ProviderError(Exception):
    """Base class for safe, client-mappable provider failures."""


class ProviderConfigurationError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class InvalidModelOutputError(ProviderError):
    pass
