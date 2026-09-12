"""Protocol/lifecycle tests; HTTP and scheduling are controlled explicitly.

Run with the repository host harness, or for isolated protocol tests:
pytest --confcutdir=tests/v2/hdbluesignin tests/v2/hdbluesignin
"""
import copy
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


TEST_DIR = Path(__file__).resolve().parent
GENERATION = TEST_DIR.parent.name
SOURCE = TEST_DIR.parents[2] / ("plugins." + GENERATION) / "hdbluesignin"
spec = importlib.util.spec_from_file_location("hdblue_core_" + GENERATION, SOURCE / "core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
KEY = "hdbmp_" + "a" * 40
NOW = datetime(2026, 9, 12, 10, 0, tzinfo=core.TZ)


def payload(checked=False, **changes):
    data = {"protocolVersion": 1, "account": {"id": "u1", "username": "alice", "nickname": "Alice", "avatarUrl": ""},
            "date": "2026-09-12", "timezone": "Asia/Shanghai", "pointName": "积分", "points": 100,
            "reward": 5 if checked else 0, "alreadyCheckedIn": checked, "performedCheckIn": checked,
            "currentStreak": 3 if checked else 2, "maxStreak": 8,
            "history": [{"date": "2026-09-11", "reward": 5, "createdAt": "2026-09-11T00:35:00Z"}]}
    data.update(changes)
    return data


class Scheduler:
    def __init__(self, **kwargs):
        self.jobs = {}
        self.running = False
        self.removed = False

    def start(self):
        self.running = True

    def add_job(self, func, trigger, **kwargs):
        self.jobs[kwargs["id"]] = SimpleNamespace(func=func, next_run_time=kwargs["run_date"], **kwargs)

    def remove_all_jobs(self):
        self.jobs.clear()
        self.removed = True

    def shutdown(self, wait):
        assert wait is False
        self.running = False

    def get_jobs(self):
        return list(self.jobs.values())


class Plugin(core.HDBluePlugin):
    notification_type = "site"
    host_logger = Mock()

    def __init__(self):
        self.storage = {}
        self.post_message = Mock()
        self.update_config = Mock()

    def get_data(self, key):
        return copy.deepcopy(self.storage.get(key))

    def save_data(self, key, value):
        self.storage[key] = copy.deepcopy(value)


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setattr(core, "BackgroundScheduler", Scheduler)
    monkeypatch.setattr(core, "now_beijing", lambda: NOW)
    monkeypatch.setattr(core.random, "randint", lambda a, b: b)
    item = Plugin()
    item.init_plugin({"api_key": KEY, "enabled": True, "catch_up": False, "notification": "all"})
    yield item
    item.stop_service()


def run(plugin, responses, mode="auto", target="2026-09-12"):
    client = Mock()
    client.request.side_effect = responses
    plugin._client = lambda: client
    plugin._run(mode, target, plugin._generation)
    return client


def test_first_checkin_reads_then_posts_and_records_actual_reward(plugin):
    client = run(plugin, [payload(), payload(True, points=127, reward=27)])
    assert client.request.call_args_list[0].args == ("GET", "/check-in")
    assert client.request.call_args_list[1].args == ("POST", "/check-in")
    assert plugin.storage["state"]["snapshot"]["reward"] == 27
    assert plugin.storage["state"]["lastResult"]["result"] == "签到成功"
    plugin.post_message.assert_called_once()


def test_already_checked_does_not_post_or_claim_new_reward(plugin):
    client = run(plugin, [payload(True)])
    assert client.request.call_count == 1
    assert plugin.storage["state"]["lastResult"]["result"] == "今日已签到"


def test_racing_web_checkin_is_reported_as_already_checked(plugin):
    run(plugin, [payload(), payload(True, performedCheckIn=False)])
    assert plugin.storage["state"]["lastResult"]["result"] == "今日已签到"


def test_post_timeout_checks_server_before_retrying(plugin):
    error = core.CheckinError("超时", retryable=True, uncertain=True)
    client = run(plugin, [payload(), error, payload(True)])
    assert [call.args[0] for call in client.request.call_args_list] == ["GET", "POST", "GET"]
    assert plugin.storage["state"]["lastResult"]["result"] == "已确认签到成功"
    assert "retry" not in plugin.storage["state"]


def test_uncertain_post_does_not_post_again_in_same_execution(plugin):
    error = core.CheckinError("超时", retryable=True, uncertain=True)
    client = run(plugin, [payload(), error, payload()])
    assert sum(call.args[0] == "POST" for call in client.request.call_args_list) == 1
    assert plugin.storage["state"]["lastResult"]["result"] == "结果待确认"
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=10)


def test_retries_are_bounded_and_survive_reload(plugin):
    error = core.CheckinError("网络故障", retryable=True)
    run(plugin, [error])
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=10)
    plugin.init_plugin(plugin._config)
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=10)
    run(plugin, [error])
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=30)
    run(plugin, [error])
    assert "retry" not in plugin._state
    client = run(plugin, [error])
    client.request.assert_not_called()
    assert plugin._state["attempts"]["count"] == 3


def test_rate_limit_honors_retry_after(plugin):
    run(plugin, [core.CheckinError("限流", retryable=True, retry_after=2700)])
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=45)


def test_auth_failure_stops_automatic_requests_until_key_test(plugin):
    run(plugin, [core.CheckinError("无权限", auth=True)])
    assert plugin._auth_blocked
    assert "retry" not in plugin._state
    assert run(plugin, [payload()]).request.call_count == 0
    client = run(plugin, [payload()], mode="test")
    assert client.request.call_args.args == ("GET", "/me")
    assert not plugin._auth_blocked


def test_connection_test_is_readonly_and_resets_toggle(plugin):
    plugin.init_plugin(dict(plugin._config, test_connection=True, run_once=True))
    assert set(plugin._scheduler.jobs) == {"test"}
    assert not plugin.update_config.call_args.args[0]["test_connection"]
    assert not plugin.update_config.call_args.args[0]["run_once"]
    client = run(plugin, [payload()], mode="test")
    assert client.request.call_count == 1
    plugin.post_message.assert_not_called()


def test_changed_credential_discards_old_account_and_retry_state(plugin):
    run(plugin, [payload(True)])
    assert plugin._state["snapshot"]["account"]["username"] == "alice"
    plugin.init_plugin(dict(plugin._config, api_key="hdbmp_" + "b" * 40))
    assert "snapshot" not in plugin.storage["state"]
    assert "events" not in plugin.storage["state"]
    assert "alice" not in json.dumps(plugin.get_page())


def test_cache_survives_reload_with_same_credential(plugin):
    run(plugin, [payload(True)])
    plugin.init_plugin(plugin._config)
    assert plugin._state["snapshot"]["account"]["username"] == "alice"
    assert run(plugin, [payload()]).request.call_count == 0


def test_stop_and_reload_invalidate_old_jobs(plugin):
    old_generation = plugin._generation
    old_scheduler = plugin._scheduler
    plugin.init_plugin(plugin._config)
    assert old_scheduler.removed and not old_scheduler.running
    assert plugin._generation != old_generation
    client = Mock()
    plugin._client = lambda: client
    plugin._run("auto", "2026-09-12", old_generation)
    client.request.assert_not_called()
    current_scheduler = plugin._scheduler
    plugin.stop_service()
    assert current_scheduler.removed and not current_scheduler.running
    plugin.scheduled_checkin()
    client.request.assert_not_called()


def test_configuration_change_during_http_prevents_post_and_cache(plugin):
    def request(method, endpoint):
        plugin.stop_service()
        return payload()
    client = Mock()
    client.request.side_effect = request
    plugin._client = lambda: client
    plugin._run("auto", "2026-09-12", plugin._generation)
    assert client.request.call_count == 1
    assert "snapshot" not in plugin.storage["state"]


def test_concurrent_callback_cannot_issue_second_request(plugin):
    plugin._run_lock.acquire()
    try:
        client = run(plugin, [payload(True)])
        client.request.assert_not_called()
    finally:
        plugin._run_lock.release()


def test_previous_day_callback_and_changed_server_date_never_post(plugin):
    assert run(plugin, [payload()], target="2026-09-11").request.call_count == 0
    client = run(plugin, [payload(date="2026-09-13")])
    assert client.request.call_count == 1
    assert "往日" in plugin._state["lastResult"]["message"]


def test_retry_never_crosses_midnight(plugin, monkeypatch):
    late = NOW.replace(hour=23, minute=58)
    monkeypatch.setattr(core, "now_beijing", lambda: late)
    run(plugin, [core.CheckinError("网络故障", retryable=True)])
    assert "retry" not in plugin._state
    assert not plugin._scheduler.jobs


def test_cron_is_in_beijing_and_catchup_only_after_due_time(plugin, monkeypatch):
    service = plugin.get_service()[0]
    assert str(service["trigger"].timezone) == "Asia/Shanghai"
    assert service["id"] == "PluginDailyCheckin"
    plugin.init_plugin(dict(plugin._config, catch_up=True))
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=30, seconds=15)
    monkeypatch.setattr(core, "now_beijing", lambda: NOW.replace(hour=7))
    plugin.init_plugin(dict(plugin._config, catch_up=True))
    assert not plugin._scheduler.jobs


def test_scheduled_jitter_and_no_duplicate_pending_retry(plugin):
    plugin.scheduled_checkin()
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=30)
    run(plugin, [core.CheckinError("网络故障", retryable=True)])
    plugin.scheduled_checkin()
    assert plugin._scheduler.jobs["checkin"].next_run_time == NOW + timedelta(minutes=10)


def test_frequent_cron_does_not_postpone_pending_jitter(plugin, monkeypatch):
    plugin.init_plugin(dict(plugin._config, cron="* * * * *"))
    plugin.scheduled_checkin()
    due = plugin._scheduler.jobs["checkin"].next_run_time
    monkeypatch.setattr(core, "now_beijing", lambda: NOW + timedelta(minutes=1))
    plugin.scheduled_checkin()
    assert plugin._scheduler.jobs["checkin"].next_run_time == due


def test_invalid_configuration_registers_no_jobs(plugin):
    plugin.init_plugin(dict(plugin._config, cron="bad cron", api_key="password"))
    assert plugin.get_service() == []
    assert plugin._scheduler is None
    assert plugin._config_error


def test_notification_failure_does_not_undo_success(plugin):
    plugin.post_message.side_effect = RuntimeError("channel unavailable")
    run(plugin, [payload(True)])
    assert plugin._state["snapshot"]["alreadyCheckedIn"]
    assert "retry" not in plugin._state


def test_stop_during_record_persistence_suppresses_notification(plugin):
    original_save = plugin.save_data
    def save(key, value):
        original_save(key, value)
        if "snapshot" in value:
            plugin.stop_service()
    plugin.save_data = save
    run(plugin, [payload(True)])
    plugin.post_message.assert_not_called()


def test_stop_during_logging_suppresses_notification(plugin):
    plugin._log = lambda text: plugin.stop_service()
    run(plugin, [payload(True)])
    plugin.post_message.assert_not_called()


def test_late_day_jitter_is_clamped_instead_of_dropping_run(plugin, monkeypatch):
    late = NOW.replace(hour=23, minute=55)
    monkeypatch.setattr(core, "now_beijing", lambda: late)
    plugin.init_plugin(dict(plugin._config, cron="55 23 * * *", catch_up=True))
    assert plugin._scheduler.jobs["checkin"].next_run_time == late.replace(minute=59, second=30)
    plugin._scheduler.remove_all_jobs()
    plugin.scheduled_checkin()
    assert plugin._scheduler.jobs["checkin"].next_run_time == late.replace(minute=59, second=30)


def test_page_and_form_never_send_requests_or_expose_key(plugin):
    client = Mock()
    plugin._client = lambda: client
    content = json.dumps([plugin.get_page(), plugin.get_form()])
    client.request.assert_not_called()
    assert KEY not in content
    assert '"type": "password"' in content


def test_form_preserves_all_models_defaults_and_input_contracts(plugin):
    form, defaults = plugin.get_form()
    cells = form[0]["content"][0]["content"]
    controls = {cell["content"][0]["props"]["model"]: cell["content"][0]
                for cell in cells if "model" in cell["content"][0].get("props", {})}
    assert defaults == {
        "enabled": False, "api_key": "", "cron": "30 8 * * *", "jitter_minutes": 30,
        "catch_up": True, "notification": "failure", "use_proxy": False, "proxy_url": "",
        "run_once": False, "test_connection": False,
    }
    assert set(controls) == set(defaults)
    assert len(controls) == sum("model" in cell["content"][0].get("props", {}) for cell in cells)
    for model in ("api_key", "proxy_url"):
        assert controls[model]["props"]["type"] == "password"
        assert controls[model]["props"]["autocomplete"] == "off"
    assert controls["jitter_minutes"]["props"]["min"] == 0
    assert controls["jitter_minutes"]["props"]["max"] == 30
    assert [item["value"] for item in controls["notification"]["props"]["items"]] == ["failure", "all", "off"]


def test_form_grid_reserves_label_hint_spacing_and_stacks_on_mobile(plugin):
    form, _ = plugin.get_form()
    assert form[0]["component"] == "VForm"
    row = form[0]["content"][0]
    assert row["component"] == "VRow"
    assert row["props"]["class"] == "ma-0"
    for cell in row["content"]:
        assert cell["component"] == "VCol"
        assert cell["props"]["cols"] == 12
        assert cell["props"]["class"] == "pa-2"
        assert len(cell["content"]) == 1
        control = cell["content"][0]
        props = control.get("props", {})
        assert cell["props"]["md"] == (6 if props.get("model") in ("cron", "jitter_minutes") else 12)
        if control["component"] in ("VTextField", "VSelect"):
            assert props["hide-details"] is False
        if props.get("hint"):
            assert props["persistent-hint"] is True


def test_form_layout_has_no_runtime_or_default_mutation(plugin):
    config = copy.deepcopy(plugin._config)
    state = copy.deepcopy(plugin._state)
    stored = copy.deepcopy(plugin.storage)
    jobs = list(plugin._scheduler.jobs)
    defaults_before = copy.deepcopy(core.DEFAULTS)
    client = Mock()
    plugin._client = lambda: client
    _, returned_defaults = plugin.get_form()
    returned_defaults["cron"] = "0 0 * * *"
    assert core.DEFAULTS == defaults_before
    assert plugin._config == config
    assert plugin._state == state
    assert plugin.storage == stored
    assert list(plugin._scheduler.jobs) == jobs
    client.request.assert_not_called()
    plugin.update_config.assert_not_called()
    plugin.post_message.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"protocolVersion": 2}, {"history": {}}, {"points": "100"}, {"date": "2026-09-99"},
    {"timezone": "UTC"}, {"alreadyCheckedIn": "false"}, {"account": {}}, {"reward": -5},
])
def test_protocol_rejects_malformed_data(changes):
    with pytest.raises(core.CheckinError):
        core.validate_payload(payload(**changes))


def test_protocol_discards_unknown_sensitive_fields():
    data = payload(token=KEY, email="private@example.com")
    data["account"]["email"] = "private@example.com"
    safe = json.dumps(core.validate_payload(data))
    assert KEY not in safe and "private@example.com" not in safe


class Session:
    def __init__(self, response):
        self.response = response
        self.trust_env = True
        self.proxies = {}
        self.request = Mock(return_value=response)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def response(status=200, data=None, headers=None):
    item = Mock(status_code=status, headers=headers or {}, content=b"{}")
    item.json.return_value = data if data is not None else {"code": 0, "data": payload()}
    return item


def test_http_fixed_origin_tls_no_redirects_no_ambient_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.invalid:8888")
    session = Session(response())
    client = core.ForumClient(KEY, session_factory=lambda: session)
    client.request("POST", "/check-in")
    call = session.request.call_args
    assert call.args == ("POST", "https://hdblue.cc/api/integrations/moviepilot/v1/check-in")
    assert call.kwargs["json"] == {}
    assert call.kwargs["verify"] is True
    assert call.kwargs["allow_redirects"] is False
    assert call.kwargs["headers"]["Authorization"] == "Bearer " + KEY
    assert session.trust_env is False and session.proxies == {}


def test_http_explicit_proxy():
    session = Session(response())
    core.ForumClient(KEY, "http://proxy:8080", lambda: session).request("GET", "/me")
    assert session.proxies == {"http": "http://proxy:8080", "https": "http://proxy:8080"}


@pytest.mark.parametrize("status,retryable,auth", [(401, False, True), (403, False, False),
    (429, True, False), (503, True, False), (302, False, False), (400, False, False)])
def test_http_errors_are_classified_without_body_or_key_leak(status, retryable, auth):
    session = Session(response(status, {"message": KEY}, {"Retry-After": "900"}))
    with pytest.raises(core.CheckinError) as caught:
        core.ForumClient(KEY, session_factory=lambda: session).request("GET", "/check-in")
    assert caught.value.retryable is retryable
    assert caught.value.auth is auth
    assert KEY not in str(caught.value)
    if status == 429:
        assert caught.value.retry_after == 900


def test_transport_exception_never_leaks_request_secret():
    session = Session(response())
    session.request.side_effect = requests.Timeout("header " + KEY)
    with pytest.raises(core.CheckinError) as caught:
        core.ForumClient(KEY, session_factory=lambda: session).request("POST", "/check-in")
    assert caught.value.uncertain and caught.value.retryable
    assert KEY not in str(caught.value)


def test_successful_post_with_invalid_response_is_uncertain():
    session = Session(response(data={"code": 0, "data": {}}))
    with pytest.raises(core.CheckinError) as caught:
        core.ForumClient(KEY, session_factory=lambda: session).request("POST", "/check-in")
    assert caught.value.uncertain


def test_retry_after_accepts_http_date_and_invalid_value():
    assert core.retry_after_seconds("Sat, 12 Sep 2026 02:15:00 GMT", NOW) == 901
    assert core.retry_after_seconds("invalid", NOW) == 0


def test_metadata_and_core_match_generations():
    import ast
    tree = ast.parse((SOURCE / "__init__.py").read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    fields = {node.targets[0].id: ast.literal_eval(node.value) for node in klass.body
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)}
    package = json.loads((TEST_DIR.parents[2] / ("package." + GENERATION + ".json")).read_text())["HDBlueSignin"]
    assert fields["plugin_version"] == package["version"]
    assert fields["plugin_icon"] == package["icon"]
    assert fields["plugin_name"] == package["name"]
    assert package["release"] is False
    opposite = "v3" if GENERATION == "v2" else "v2"
    assert (SOURCE / "core.py").read_bytes() == (TEST_DIR.parents[2] / ("plugins." + opposite) / "hdbluesignin/core.py").read_bytes()
