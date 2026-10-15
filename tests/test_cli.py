from main import main, PASSES

import pytest


def test_list_passes(capsys):
    main(["list-passes"])
    out = capsys.readouterr().out
    assert all(name in out for name in PASSES)


def test_missing_model(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["visualize", "--model", str(tmp_path / "missing.xml")])
    assert e.value.code == 2


def test_unknown_pass(tiny_model_path):
    with pytest.raises(SystemExit) as e:
        main(["visualize", "--model", str(tiny_model_path), "--passes", "nope"])
    assert e.value.code == 2


# TODO end to end with tiny ollama and truncation pass or proxy
def test_visualize_end_to_end(tiny_model_path, tmp_path):
    out = tmp_path / "graph.svg"
    main(["visualize", "--model", str(tiny_model_path), "--output", str(out),
          "--passes", ",".join(PASSES), "--highlight-names", "mm"])
    assert out.exists()
    assert "<svg" in out.read_text(encoding="utf-8") # validate open tag
