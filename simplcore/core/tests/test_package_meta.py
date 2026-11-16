import simplcore


def test_package_metadata_and_env_report() -> None:
    assert simplcore.__version__
    report = simplcore.report_environment()
    assert report["project"] == "SimplCore"
    assert "cache_root" in report
