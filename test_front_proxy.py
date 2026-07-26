import front_proxy


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
