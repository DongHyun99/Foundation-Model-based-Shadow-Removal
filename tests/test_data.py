import numpy as np
from PIL import Image

from fmsr.config import load_config
from fmsr.data import image_key, index_by_key, list_images, load_mask_bin, output_name


def test_filename_conventions():
    assert output_name("_MG_5580.JPG.jpg") == "_MG_5580.JPG.png"
    assert image_key("_MG_5580.JPG.jpg") == "_MG_5580"
    assert image_key("_MG_5580.JPG.png") == "_MG_5580"
    assert output_name("100-1.png") == "100-1.png"


def test_list_images_skips_dotfiles_and_non_images(tmp_path):
    for name in ["b.png", "a.jpg", ".MATLABDriveTag", "notes.txt", ".hidden.png"]:
        (tmp_path / name).write_bytes(b"x")
    assert list_images(tmp_path) == ["a.jpg", "b.png"]


def test_index_by_key_matches_pred_and_gt_names(tmp_path):
    (tmp_path / "_MG_5580.JPG.jpg").write_bytes(b"x")
    assert index_by_key(tmp_path) == {"_MG_5580": "_MG_5580.JPG.jpg"}


def test_mask_binarization_threshold(tmp_path):
    gray = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    Image.fromarray(gray, mode="L").save(tmp_path / "m.png")
    m = np.asarray(load_mask_bin(tmp_path / "m.png", 127))
    assert m.tolist() == [[0, 0], [255, 255]]


def test_configs_load_and_match_dataset_counts():
    cfg = load_config()
    assert list(cfg.models.variants) == [
        "FLUX.1-Fill-dev",
        "FLUX.1-Kontext-dev",
        "Qwen-Image-Inpaint",
        "Qwen-Image-Edit-2509",
        "FLUX.1-Kontext-dev-Mask",
        "Qwen-Image-Edit-2509-Mask",
    ]
    assert [r.prompt_id for r in cfg.runs("Qwen-Image-Edit-2509-Mask")] == ["m1", "m2", "m3", "m4"]
    assert {r.sampling.id for r in cfg.runs("FLUX.1-Kontext-dev-Mask")} == {"default"}
    assert cfg.prompts["inpaint"]["i0"] == ""
    assert len(cfg.runs("FLUX.1-Fill-dev")) == 4
    assert [r.prompt_id for r in cfg.runs("Qwen-Image-Edit-2509")] == ["e1", "e2", "e3", "e4"]
    assert cfg.samplings["default"].params("Qwen-Image-Edit-2509")["vae_image_area"] == 256 * 256
    assert cfg.samplings["default"].params("FLUX.1-Kontext-dev")["max_area"] == 256 * 256
    for test, n in cfg.datasets.tests.items():
        for part in ("shadow", "mask", "gt"):
            assert len(list_images(cfg.datasets.dir(test, part))) == n, (test, part)
