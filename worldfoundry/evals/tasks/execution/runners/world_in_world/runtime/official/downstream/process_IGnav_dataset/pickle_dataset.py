# igdataset_pack.py
import gzip
import io
import json
import os.path as osp
import pickle
import zipfile
from typing import Any

from downstream.downstream_datasets import IGDataset
from PIL import Image

PORTABLE_VERSION = "1.1"


def _encode_image_as_png_bytes(img_path: str) -> bytes:
    """Load the image and re-encode it as PNG bytes for portability."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _to_portable_item(item: dict[str, Any], hm3d_root: str, img_zip_relpath: str, png_bytes: bytes) -> dict[str, Any]:
    full_scene_path = item["scene_id"]
    rel_scene_path = osp.relpath(full_scene_path, hm3d_root).replace("\\", "/")

    portable = dict(item)
    portable.pop("scene_id")
    portable.pop("goal_image_path")

    portable["scene_relpath"] = rel_scene_path
    portable["scene_relpath"] = rel_scene_path
    portable["goal_image_relpath"] = img_zip_relpath
    portable["goal_image_png_bytes"] = png_bytes  # you asked to embed it in pickle
    return portable


def _from_portable_item(item: dict[str, Any], hm3d_root: str, unzipped_img_root: str) -> dict[str, Any]:
    rebuilt = dict(item)
    rel_scene_path = rebuilt.pop("scene_relpath")
    rebuilt["scene_id"] = osp.join(hm3d_root, rel_scene_path).replace("\\", "/")

    goal_rel = rebuilt.pop("goal_image_relpath")
    rebuilt["goal_image_path"] = osp.join(unzipped_img_root, goal_rel).replace("\\", "/")

    # keep the bytes as well (can be large, you may delete if you do not need in RAM)
    return rebuilt


def save_igdataset(
    ds: IGDataset,
    out_pkl_path: str,
    hm3d_root: str,
    out_img_zip_path: str,
    compress_pickle: bool = True,
    zip_compression=zipfile.ZIP_DEFLATED,
) -> None:
    """
    Copy all goal images into a zip and also store their PNG bytes in the pickle.
    The goal_image_path in each datum is rewritten to a relative path inside the zip.
    """

    # Prepare zip
    zf = zipfile.ZipFile(out_img_zip_path, mode="w", compression=zip_compression)

    portable_data = []
    for sample in ds.data:
        datum_id = sample["datum_id"]
        # Put images under images/<datum_id>.png
        zip_rel = f"images/{datum_id}.png"

        img_bytes = _encode_image_as_png_bytes(sample["goal_image_path"])
        zf.writestr(zip_rel, img_bytes)

        portable = _to_portable_item(
            sample,
            hm3d_root=hm3d_root,
            img_zip_relpath=zip_rel,
            png_bytes=None,
        )
        portable_data.append(portable)

    zf.close()

    blob = {
        "version": PORTABLE_VERSION,
        "num_items": len(ds),
        "meta": {"original_hm3d_root": hm3d_root, "img_zip_filename": osp.basename(out_img_zip_path)},
        "data": portable_data,
    }

    opener = gzip.open if compress_pickle else open
    with opener(out_pkl_path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved IGDataset to {out_pkl_path}")


class IGDatasetPortable(IGDataset):
    """
    A thin wrapper that returns the same dictionaries but now they also contain
    goal_image_png_bytes. __len__ and __getitem__ work as usual.
    """

    def __init__(self, rebuilt_data: list[dict[str, Any]]):
        self.data = rebuilt_data

        print("IGDataset")
        print(f"Total kept   : {len(self.data)}")

    # __len__ and __getitem__ inherited from IGDataset


def load_igdataset(
    pkl_path: str,
    hm3d_root: str,
    unzipped_img_root: str,
) -> IGDatasetPortable:
    # Determine if it's JSON or pickle based on file extension
    is_json = pkl_path.endswith((".json", ".json.gz"))

    if is_json:
        # JSON files: use text mode
        opener = gzip.open if pkl_path.endswith(".gz") else open
        with opener(pkl_path, "rt", encoding="utf-8") as f:
            blob = json.load(f)
    else:
        # Pickle files: use binary mode
        opener = gzip.open if pkl_path.endswith(".gz") else open
        with opener(pkl_path, "rb") as f:
            blob = pickle.load(f)

    rebuilt_data = [_from_portable_item(x, hm3d_root, unzipped_img_root) for x in blob["data"]]
    ds = IGDatasetPortable(rebuilt_data)
    return ds


def decode_png_bytes(png_bytes: bytes) -> Image.Image:
    """Helper to reconstruct a PIL.Image from the stored bytes."""
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def load_igdataset_from_zip(
    zip_path: str,
    pkl_path: str,
    remote_hm3d_root: str = "data/scene_datasets/hm3d/val",
    remote_img_folder: str = "goal_imgs_unzipped",
) -> IGDatasetPortable:
    remote_img_path = osp.join(osp.dirname(zip_path), remote_img_folder)
    # unzip once
    if not osp.exists(remote_img_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(remote_img_path)

    ds = load_igdataset(pkl_path, hm3d_root=remote_hm3d_root, unzipped_img_root=remote_img_path)

    print(f"Loaded IGDataset from path: {pkl_path}, goal images unzipped to: {remote_img_path}")
    return ds
