"""
Downloading single cell Cell Painting profiles from the Cell Health experiment

Note, together the files are 130 GB, and downloading will take a long time.

See https://github.com/broadinstitute/cell-health/blob/master/0.download-data/download.py
"""

import pathlib
import requests
from concurrent.futures import ThreadPoolExecutor


def download_sqllite_file(plate, file_info, download_dir):
    figshare_id = file_info[plate]
    filename = pathlib.Path(download_dir, f"{plate}.sqlite")
    url = f"https://nih.figshare.com/ndownloader/files/{figshare_id}"

    if not filename.exists():
        print(f"Now downloading... {filename}")
        with requests.get(url, stream=True) as sql_request:
            sql_request.raise_for_status()
            with open(filename, 'wb') as sql_fh:
                for chunk in sql_request.iter_content(chunk_size=819200000):
                    if chunk:
                        sql_fh.write(chunk)
        print(f"Done... {filename}\n")

file_info = {
    "SQ00014610": "18028784",
    "SQ00014611": "18508583",
    "SQ00014612": "18505937",
    "SQ00014613": "18506036",
    "SQ00014614": "18031619",
    "SQ00014615": "18506108",
    "SQ00014616": "18506912",
    "SQ00014617": "18508316",
    "SQ00014618": "18508421",
}

download_dir = pathlib.Path("data/cell_health")
download_dir.mkdir(exist_ok=True, parents=True)

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(download_sqllite_file, plate, file_info, download_dir) for plate in file_info]

