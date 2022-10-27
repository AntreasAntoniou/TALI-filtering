import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pathlib
import tqdm
from tabulate import tabulate
from rich import print
import shutil

source_table_filepath = pathlib.Path("/data/datasets/tali-wit-2-1/all/")
target_table_filepath = pathlib.Path("/data/datasets/tali-wit-2-1-buckets/all/")

dir_idx_list = [
    subdir.name
    for subdir in tqdm.tqdm(source_table_filepath.iterdir())
    if subdir.is_dir()
]

bucket_size = 1000

sorted_dir_idx_list = sorted(dir_idx_list, key=int)
with tqdm.tqdm(total=len(sorted_dir_idx_list)) as pbar:
    for dir_idx in sorted_dir_idx_list:
        source_path = source_table_filepath / dir_idx
        bucket_idx = int(dir_idx) // bucket_size
        target_path = target_table_filepath / f"{bucket_idx}" / dir_idx
        shutil.move(source_path, target_path)
        pbar.update(1)
