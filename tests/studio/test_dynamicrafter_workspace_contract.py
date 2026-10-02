from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry


def test_dynamicrafter_1024_uses_downloaded_doubiiu_checkpoint() -> None:
    entry = find_entry("dynamicrafter_1024_i2v")

    assert Path(entry.default_model_ref).name == "model.ckpt"
    assert Path(entry.default_model_ref).parent.name == "Doubiiu--DynamiCrafter_1024"
    assert Path(entry.default_model_ref).is_file()
    assert Path(entry.default_input_path).is_file()
