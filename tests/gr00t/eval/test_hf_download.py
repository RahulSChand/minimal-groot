from unittest.mock import Mock

import pytest
import requests

from gr00t.eval import hf_download as module


@pytest.fixture
def download(monkeypatch):
    fetch, sleep = Mock(return_value="cached"), Mock()
    monkeypatch.setattr(module, "snapshot_download", fetch)
    monkeypatch.setattr(module.time, "sleep", sleep)
    monkeypatch.setattr(module.random, "uniform", lambda *_: 0)
    return fetch, sleep


def run(**kwargs):
    return module.download_checkpoint("user/repo", "epoch-007", "a" * 40, "/cache", **kwargs)


def test_timeout_then_success(download):
    fetch, sleep = download
    fetch.side_effect = [requests.ReadTimeout(), "cached"]
    assert run() == "cached"
    sleep.assert_called_once_with(30)
    assert fetch.call_args.kwargs["max_workers"] == 1
    assert fetch.call_args.kwargs["local_dir"] == "/cache"
    assert "force_download" not in fetch.call_args.kwargs


def test_bounded_retries_keep_final_cooldown(download):
    fetch, sleep = download
    fetch.side_effect = requests.ConnectionError()
    with pytest.raises(module.DownloadDeferred):
        run()
    assert fetch.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [30, 60, 120, 240, 300, 300]


@pytest.mark.parametrize("status", [401, 403, 404, 429, 503])
def test_http_failures(download, status):
    fetch, sleep = download
    response = requests.Response()
    response.status_code = status
    response.headers["Retry-After"] = "900"
    fetch.side_effect = requests.HTTPError(response=response)
    with pytest.raises(module.DownloadDeferred if status >= 429 else requests.HTTPError):
        run(attempts=1)
    if status >= 429:
        sleep.assert_called_once_with(900)
    else:
        sleep.assert_not_called()
