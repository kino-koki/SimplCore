from pathlib import Path


def test_java_high_performance_pom_targets_java26():
    root = Path(__file__).resolve().parents[3]
    pom = root / "java-high-performance" / "pom.xml"
    text = pom.read_text(encoding="utf-8")

    assert "<java.version>26</java.version>" in text
    assert "using Java 26" in text
