from collections import defaultdict
import pathlib
from typing import DefaultDict
from torch.utils.data import Dataset
import pyarrow as pa
from dataclasses import dataclass
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from rich import print
from rich.traceback import install
from tqdm import tqdm
from transformers import data

install()


@dataclass
class TALISchema:
    wit_idx: pa.int64()
    term_idx: pa.int64()
    sort_type: pa.string()
    age_restricted: pa.bool_()
    author: pa.string()
    channel_id: pa.string()
    channel_url: pa.string()
    description: pa.string()
    embed_url: pa.string()
    keywords: pa.list_(pa.string())
    length: pa.int64()
    publish_date: pa.timestamp("us")
    thumbnail_url: pa.string()
    title: pa.string()
    video_id: pa.string()
    video_store_filepath: pa.string()
    views: pa.string()
    watch_url: pa.string()


tali_schema = list(TALISchema.__dict__["__annotations__"].items())
tali_schema = pa.schema(tali_schema)


class TALIDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = pathlib.Path(data_root)
        self.captions_table_root = self.data_root / "captions.parquet" / "relevance"
        self.video_paths_table_root = (
            self.data_root / "wit_to_video_paths.parquet" / "relevance"
        )
        self.score_table_root = self.data_root / "clip_scores.parquet" / "relevance"
        self.video_data_root = self.data_root / "video_data.parquet"

        self.total_samples = 0
        self.dataset_idx_to_bucket_idx_and_item_idx = []
        with tqdm(total=400) as pbar:
            for bucket_idx in range(400):
                bucket_path = self.video_paths_table_root / f"{bucket_idx}"
                pbar.update(1)
                pbar.set_description(f"{self.total_samples}")
                if not bucket_path.exists():
                    continue
                bucket_table = ds.dataset(bucket_path, schema=tali_schema).to_table()
                bucket_idx_and_item_idx = [
                    (bucket_idx, i) for i in range(len(bucket_table))
                ]
                self.dataset_idx_to_bucket_idx_and_item_idx.extend(
                    bucket_idx_and_item_idx
                )

                self.total_samples += bucket_table.num_rows

        print(f"Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    dataset = TALIDataset(data_root="/data/datasets/tali-wit-2-1-buckets")
