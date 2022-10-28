import pathlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from math import floor
from typing import DefaultDict

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from rich import print
from rich.traceback import install
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from tqdm import tqdm

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


def trim_video(video_path, start_sec, end_sec, output_path):
    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(timedelta(seconds=start_sec)),
            "-to",
            str(timedelta(seconds=end_sec)),
            "-i",
            video_path,
            "-c",
            "copy",
            output_path,
        ]
    )


def extract_frames_from_video(video_path: str):
    video: EncodedVideo = EncodedVideo.from_path(video_path)

    clip_start_sec = 0.0  # secs
    video_duration_seconds = float(video.duration)

    video_data = video.get_clip(
        start_sec=clip_start_sec, end_sec=clip_start_sec + video_duration_seconds
    )

    video_sample_rate = floor(video_data["video"].shape[1] / video_duration_seconds)
    num_frames = video_duration_seconds

    video_data["video"] = video_data["video"][
        :, : video_sample_rate * floor(video_duration_seconds)
    ]

    # Compose video data transforms
    video_transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_samples=floor(num_frames), temporal_dim=1),
                Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                CenterCropVideo(crop_size=(224, 224)),
            ]
        ),
    )

    video_data = video_transform(video_data)
    return video_data


def filter_dataset(data_root: str, clip_name_or_path: str):
    data_root = pathlib.Path(data_root)
    captions_table_root = data_root / "captions.parquet" / "relevance"
    video_paths_table_root = data_root / "wit_to_video_paths.parquet" / "relevance"
    score_table_root = data_root / "clip_scores.parquet" / "relevance"
    video_data_root = data_root / "video_data.parquet"

    total_samples = 0
    dataset_idx_to_bucket_idx_and_item_idx = []
    with tqdm(total=400) as pbar:
        for bucket_idx in range(400):
            bucket_path = video_paths_table_root / f"{bucket_idx}"
            pbar.update(1)
            pbar.set_description(f"{total_samples}")
            if not bucket_path.exists():
                continue
            bucket_table = ds.dataset(bucket_path, schema=tali_schema).to_table()
            bucket_idx_and_item_idx = [
                (bucket_idx, i) for i in range(len(bucket_table))
            ]
            dataset_idx_to_bucket_idx_and_item_idx.extend(bucket_idx_and_item_idx)

            total_samples += bucket_table.num_rows
