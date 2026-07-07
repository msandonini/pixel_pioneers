from __future__ import annotations

import pipeline.config as config
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from rarfile import RarFile


def download_file(url: str, dst: Path) -> None:
    """
    Downloads a file from a url to a specified destination
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[download] skip - {dst.name} already exists")
        return
    
    print(f"[download] {url}")
    with urllib.request.urlopen(url) as response, open(dst, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def extract_archive(archive_path: Path, extract_to: Path):
    """
    Extracts a zip file to a specified destination
    """

    print(f"[extract] {archive_path.name}")

    suff = archive_path.suffix.lower()

    if suff == ".rar":
        with RarFile(archive_path, "r") as archive:
            archive.extractall(extract_to)
        return
    
    with ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_to)


def download_dataset(cache_root: str | Path, 
                   dataset_name: str,
                   data_urls: dict,
                   clean: bool = False) -> Path:
    """
    Downloads a dataset from a set of urls to the specified cache directory
    """
    cache_root = Path(cache_root)
    dataset_root = cache_root / dataset_name
    archives_dir = dataset_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    for key, url in data_urls.items():
        archive_name = url.split("/")[-1]
        archive_path = archives_dir / archive_name
        download_file(url, archive_path)
        extract_archive(archive_path, dataset_root / "extracted")

        if clean:
            archive_path.unlink(missing_ok=True)
    
    print(f"[cache] {dataset_name} ready at: {dataset_root}")
    return dataset_root / "extracted"


def download_datasets(conf: dict = config.parse_args()):
    if "data_cache" in conf:
        data_cache = conf["data_cache"]
    else:
        data_cache = config.DEFAULT_CONFIG["data_cache"]
    
    if "datasets" in conf:
        datasets = conf["datasets"]
    else:
        datasets = config.DEFAULT_CONFIG["datasets"]
    
    data_storage: dict[str, Path] = {}
    for key, value in datasets.items():
        urls = []
        if "urls" in value:
            urls = value["urls"]
        clean = False
        if "clean" in value:
            clean = value["clean"]
        
        data_storage[key] = download_dataset(
            cache_root=data_cache, 
            dataset_name=key,
            data_urls=urls,
            clean=clean)
    
    return data_storage
