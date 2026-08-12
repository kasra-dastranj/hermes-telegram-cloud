import front_proxy


def test_configured_webhook_path_preserves_namespace():
    assert (
        front_proxy.configured_webhook_path(
            "https://example.test/hermes/telegram"
        )
        == "/hermes/telegram"
    )


def test_configured_webhook_path_defaults_to_telegram():
    assert front_proxy.configured_webhook_path("") == "/telegram"


def test_namespaced_health_path_maps_to_local_health():
    assert (
        front_proxy.local_public_path(
            "/hermes/health", "/hermes/telegram"
        )
        == "/health"
    )


def test_component_health_requires_every_internal_service(monkeypatch):
    ready_ports = {
        front_proxy.HERMES_PORT,
        front_proxy.ROUTER_PORT,
        front_proxy.MODEL_PROXY_PORT,
    }
    monkeypatch.setattr(
        front_proxy, "port_ready", lambda port: port in ready_ports
    )

    assert all(front_proxy.component_health().values())

    ready_ports.remove(front_proxy.MODEL_PROXY_PORT)
    health = front_proxy.component_health()
    assert health["hermes"] is True
    assert health["router"] is True
    assert health["model_proxy"] is False
