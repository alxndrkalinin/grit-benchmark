import pathlib
import concurrent.futures

import pandas as pd

from pycytominer import normalize
from pycytominer.cyto_utils import cells, output

plates = [
    "SQ00014610",
    "SQ00014611",
    "SQ00014612",
    "SQ00014613",
    "SQ00014614",
    "SQ00014615",
    "SQ00014616",
    "SQ00014617",
    "SQ00014618",
]

compression_options = {"method": "gzip", "mtime": 1}

# Set output directory where normalized single cell files will live
data_dir = pathlib.Path("../../data/cell_health/")
output_dir = pathlib.Path(data_dir / "normalized")
output_dir.mkdir(exist_ok=True, parents=True)

# Load Metadata on plate info
metadata_file = "https://github.com/broadinstitute/cell-health/blob/cd91bd0daacef2b5ea25dcceb62482bb664d9de1/1.generate-profiles/data/metadata/platemap/DEPENDENCIES1_ES2.csv?raw=True"
metadata_df = pd.read_csv(metadata_file)

metadata_df.columns = [f"Metadata_{x}" for x in metadata_df.columns]

print(metadata_df.shape)
metadata_df.head()


def process_plate(plate, output_dir, metadata_df):
    sql_file = f"sqlite:///{data_dir / plate}.sqlite"
    output_file = pathlib.Path(output_dir, f"{plate}_normalized.csv.gz")

    print(f"Now processing... {output_file}")

    sc = cells.SingleCells(
        sql_file=sql_file,
        strata=["Image_Metadata_Plate", "Image_Metadata_Well"],
        load_image_data=False
    )

    sc_df = sc.merge_single_cells()
    sc_df = normalize(profiles=sc_df, method="standardize")

    sc_df = (
        sc.image_df.merge(
            metadata_df,
            left_on="Image_Metadata_Well",
            right_on="Metadata_well_position",
            how="left",
        )
        .merge(
            sc_df,
            left_on=["TableNumber", "ImageNumber"],
            right_on=["Metadata_TableNumber", "Metadata_ImageNumber"],
            how="right",
        )
        .drop(
            [
                "TableNumber",
                "ImageNumber",
                "Metadata_WellRow",
                "Metadata_WellCol",
                "Metadata_well_position",
            ],
            axis="columns",
        )
        .rename(
            {
                "Image_Metadata_Plate": "Metadata_Plate",
                "Image_Metadata_Well": "Metadata_Well",
            },
            axis="columns",
        )
    )

    print(sc_df.shape)
    output(
        df=sc_df,
        output_filename=output_file,
        sep=",",
        float_format="%.5f",
        compression_options=compression_options,
    )

    print("Done.\n")
    return f"{plate} processed"

def normalize_plates(plates, output_dir, metadata_df):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_plate, plate, output_dir, metadata_df) for plate in plates]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


# process_plate(plates[0], output_dir, metadata_df)
normalize_plates(plates, output_dir, metadata_df)