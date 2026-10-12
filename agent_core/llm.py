"""The one place that talks to the Claude API.

Everything else depends on the small `create(**params)` interface, so tests
can swap in a scripted fake and the runner never imports the SDK.

Any failure that means "Claude can't be used for this run" (no key, network
down, rate limit after retries, a refusal) is converted to LLMUnavailable,
which the runner handles by switching to its deterministic playbook.
"""
from __future__ import annotations

import anthropic

# Server-side refusal fallback: if Claude Opus 5.5's safety classifiers decline
# a request, the API re-runs it on Anthropic's recommended fallback model
# inside the same call instead of returning a refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(Exception):
    """Claude cannot be used for this run; callers should degrade gracefully."""


class ClaudeLLM:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        use_fallbacks: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks
        self.available = True
        # api_key=None lets the SDK resolve credentials from the environment.
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)

    def create(self, **params):
        """messages.create with our model/effort defaults. Raises LLMUnavailable."""
        if not self.available:
            raise LLMUnavailable("Claude API already failed earlier in this run")
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            **params,
        }
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise self._fail("the API key was rejected") from e
        except anthropic.PermissionDeniedError as e:
            raise self._fail("this API key has no access to the model") from e
        except anthropic.BadRequestError as e:
            if self.use_fallbacks and "fallback" in str(e).lower():
                # Account without the fallback beta: retry once without it.
                self.use_fallbacks = False
                return self.create(**params)
            raise self._fail(f"request rejected: {e.message}", fatal=False) from e
        except anthropic.RateLimitError as e:
            raise self._fail("rate limited (after retries)", fatal=False) from e
        except anthropic.APIStatusError as e:
            raise self._fail(f"API error {e.status_code}", fatal=False) from e
        except anthropic.APIConnectionError as e:  # includes timeouts
            raise self._fail("could not reach the Claude API") from e
        except TypeError as e:
            # The SDK raises TypeError when no credentials are configured at all.
            if "authentication" in str(e).lower():
                raise self._fail("no API key configured") from e
            raise

        if response.stop_reason == "refusal":
            raise self._fail("the model declined this request", fatal=False)
        return response

    def _fail(self, reason: str, fatal: bool = True) -> LLMUnavailable:
        # Fatal = no request can succeed this run (bad key, no network). A single
        # rejected request (e.g. one web-search sub-call) does not disable the rest.
        if fatal:
            self.available = False
        return LLMUnavailable(reason)


class OfflineLLM:
    """Stand-in used when the user chooses offline mode or has no API key."""

    available = False

    def create(self, **params):
        raise LLMUnavailable("offline mode (no Claude API key)")
