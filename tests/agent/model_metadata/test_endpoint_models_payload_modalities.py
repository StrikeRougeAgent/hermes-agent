"""`/models` items that carry list-valued ``modalities`` beside ``pricing`` (#158208).

``_extract_pricing`` walks every nested dict of a ``/models`` item and compared each pricing
alias value against the ``{None, ""}`` set literal. A provider that reuses the alias word
``input``/``output`` inside ``modalities`` — ``{"input": ["text", "image"]}`` — made that
comparison raise ``TypeError: cannot use 'list' as a set element``.
``fetch_endpoint_model_metadata`` swallowed the exception for the WHOLE payload and returned
``{}``, so ``get_pricing_entry`` answered ``None`` and ``PromptCacheWarmer._decide`` stopped
with "cache economics unavailable", which also kept cache-edge compaction from ever coming
due.
"""

from unittest.mock import MagicMock, patch

import pytest

# Exactly the key ordering of the reported payload: the nested walker visits dicts in
# insertion order, so ``modalities`` must precede ``pricing`` to hit the bad comparison first.
_PRICING = {
    "prompt": "0.0000025",
    "completion": "0.00001",
    "input_cache_read": "0.00000025",
    "input_cache_write": "0.000003125",
}
_MODEL_WITH_MODALITIES = {
    "id": "custom-model",
    "object": "model",
    "context_length": 128_000,
    "modalities": {"input": ["text", "image"], "output": ["text"]},
    "pricing": dict(_PRICING),
}
_MODEL_WITHOUT_MODALITIES = {
    "id": "custom-model",
    "object": "model",
    "context_length": 128_000,
    "pricing": dict(_PRICING),
}


def _streamed(response):
    """A ``model_metadata_http.stream`` result: a context manager yielding *response*."""
    ctx = MagicMock()
    ctx.__enter__.return_value = response
    return ctx


def test_extract_pricing_skips_modality_lists_instead_of_raising():
    import agent.model_metadata as mm

    pricing = mm._extract_pricing(dict(_MODEL_WITH_MODALITIES))

    assert pricing == {
        "prompt": "0.0000025",
        "completion": "0.00001",
        "cache_read": "0.00000025",
        "cache_write": "0.000003125",
    }


def test_parse_models_payload_keeps_the_priced_entry():
    import agent.model_metadata as mm

    cache = mm._parse_models_payload({"object": "list", "data": [dict(_MODEL_WITH_MODALITIES)]})

    assert cache["custom-model"]["context_length"] == 128_000
    assert cache["custom-model"]["pricing"]["prompt"] == "0.0000025"
    assert cache["custom-model"]["pricing"]["cache_write"] == "0.000003125"


def test_a_structured_non_rate_value_is_not_a_price():
    """A nested ``{"input": {"max": 100}}`` block is not a rate: skip that alias, don't raise."""
    import agent.model_metadata as mm

    model = {"id": "weird", "limits": {"input": {"max": 100}}, "pricing": {"prompt": "0.000001"}}

    entry = mm._endpoint_model_entry(model, "weird", None)

    assert entry["pricing"] == {"prompt": "0.000001"}


def test_modalities_payload_prices_the_same_as_the_control(monkeypatch):
    """Through the real ``/models`` probe and ``get_pricing_entry``: the symptom is gone."""
    import agent.model_metadata as mm
    from agent import usage_pricing as up

    monkeypatch.setattr(mm, "_endpoint_disk_cache_put", lambda *a, **k: None)
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    base_url = "https://custom.example/v1"

    def _probe(model):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"object": "list", "data": [model]}
        with patch("agent.model_metadata.model_metadata_http.stream", return_value=_streamed(response)), patch(
            "agent.model_metadata.detect_local_server_type", return_value=None
        ):
            return mm.fetch_endpoint_model_metadata(base_url, api_key="k", force_refresh=True)

    with_modalities = _probe(dict(_MODEL_WITH_MODALITIES))
    without_modalities = _probe(dict(_MODEL_WITHOUT_MODALITIES))

    assert with_modalities["custom-model"]["pricing"]["prompt"] == "0.0000025"
    assert with_modalities == without_modalities  # the unstructured control is unchanged
    entry = up._pricing_entry_from_metadata(
        with_modalities, "custom-model", source_url=f"{base_url}/models",
        pricing_version="openai-compatible-models-api",
    )
    assert entry is not None
    input_per_million = entry.input_cost_per_million
    cache_write_per_million = entry.cache_write_cost_per_million
    assert input_per_million is not None and cache_write_per_million is not None
    assert float(str(input_per_million)) == pytest.approx(2.5)
    assert float(str(cache_write_per_million)) == pytest.approx(3.125)
