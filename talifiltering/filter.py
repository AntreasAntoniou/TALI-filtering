from enum import Enum
import os
import pathlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from math import floor

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from typing import List
from rich import print
from rich.traceback import install
import argparse
import tabulate
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo
from tqdm import tqdm
from utils import get_logger
from talifiltering.clip_utils import CLIPScoreOutput, get_scores

install()


logger = get_logger()


class ResponseTypes(Enum):
    DONE = 0
    EXISTS = 1
    FAILED = 2


@dataclass
class VideoCLIPScoreSchema:
    wit_idx: pa.int32()
    term_idx: pa.int32()
    video_id: pa.string()
    filepath: pa.string()
    reference_text: pa.string()
    scores_sorted_idx: pa.list_(pa.int32())
    scores_sorted: pa.list_(pa.float32())


video_score_schema = list(VideoCLIPScoreSchema.__dict__["__annotations__"].items())
video_score_schema = pa.schema(video_score_schema)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--subclip_length_in_seconds", default=30, type=int)
parser.add_argument("--num_subclips", type=int, default=10)
args = parser.parse_args()

data_root = pathlib.Path(args.data_root)
video_clip_scores_table_root = data_root / "video_clip_scores.parquet" / "relevance"

if not video_clip_scores_table_root.exists():
    video_clip_scores_table_root.mkdir(parents=True)


video_paths_table_root = data_root / "wit_to_video_paths.parquet" / "relevance"
captions_table_root = data_root / "captions.parquet" / "relevance"
youtube_titles_clip_scores = data_root / "clip_scores.parquet" / "relevance"
video_data_table_root = data_root / "video_data.parquet"


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


def trim_video(video_path, start_sec, end_sec, output_path) -> bool:
    try:
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
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
        return True
    except Exception as e:
        logger.exception(e)
        return False


def extract_frames_from_video(video_path: str):
    video: EncodedVideo = EncodedVideo.from_path(video_path)

    video_duration_seconds = float(video.duration)
    # print(f"Video duration: {video_duration_seconds} seconds")

    video_frames = []

    for start_sec in range(0, floor(video_duration_seconds), 30):
        video_data = video.get_clip(start_sec=start_sec, end_sec=start_sec + 0.1)[
            "video"
        ]

        video_frames.append(video_data[:, 0, :, :])

    video_data = dict(video=torch.stack(video_frames, dim=1))

    # Compose video data transforms
    video_transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                # Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                CenterCropVideo(crop_size=(224, 224)),
            ]
        ),
    )
    video_data = video_transform(video_data)
    return video_data["video"].permute(1, 0, 2, 3)


def add_row_to_table(
    wit_idx: int,
    term_idx: int,
    video_id: str,
    filepath: str,
    reference_text: str,
    scores_sorted_idx: List[int],
    scores_sorted: List[float],
) -> int:
    try:
        table_filepath = video_clip_scores_table_root / f"{video_id}.parquet"

        # if table_filepath.exists():
        #     return ResponseTypes.EXISTS
        # print(
        #     [
        #         [wit_idx],
        #         [term_idx],
        #         [video_id],
        #         [str(filepath)],
        #         [reference_text],
        #         [scores_sorted_idx],
        #         [scores_sorted],
        #     ]
        # )
        table_entry = pa.table(
            [
                [wit_idx],
                [term_idx],
                [video_id],
                [str(filepath)],
                [reference_text[0]],
                [scores_sorted_idx],
                [scores_sorted],
            ],
            schema=video_score_schema,
        )

        pq.write_table(table=table_entry, where=table_filepath)

        return ResponseTypes.DONE
    except Exception as e:
        logger.exception(e)
        return ResponseTypes.FAILED


@dataclass
class CLIPScores:
    sorted_scores_idx: List[int]
    scores_sorted: List[float]


def get_clip_scores(
    video_frames: torch.Tensor,
    reference_text: str,
):
    output_scores: CLIPScoreOutput = get_scores(
        image_list=video_frames, reference_text=reference_text
    )
    scores = output_scores.scores
    sorted_scores_idx = output_scores.sorted_scores_idx
    scores_sorted = scores[0][sorted_scores_idx]

    return CLIPScores(
        sorted_scores_idx=sorted_scores_idx,
        scores_sorted=scores_sorted,
    )


def generate_subclips(clip_scores: CLIPScores, video_id: str, video_path: str):
    current_second = 0.0
    for starting_second in sorted(clip_scores.sorted_scores_idx, reverse=False):
        if current_second < starting_second:
            ending_second = starting_second + args.subclip_length_in_seconds
            output_path = (
                video_clip_scores_table_root / video_id / f"{starting_second}.mp4"
            )
            trim_video(
                video_path=video_path,
                start_sec=starting_second,
                end_sec=ending_second,
                output_path=output_path,
            )


def collect_video_paths():

    total_samples = 0
    dataset_idx_to_bucket_idx_and_item_idx = []
    with tqdm(total=400) as pbar:
        for bucket_idx in range(400):
            bucket_path = video_paths_table_root / f"{bucket_idx}"
            pbar.update(1)

            if not bucket_path.exists():
                continue
            bucket_table = ds.dataset(bucket_path, schema=tali_schema).to_table()

            bucket_idx_and_item_idx = [
                (bucket_idx, i) for i in range(len(bucket_table))
            ]
            dataset_idx_to_bucket_idx_and_item_idx.extend(bucket_idx_and_item_idx)

            total_samples += bucket_table.num_rows
            pbar.set_description(f"{total_samples}")
    return dataset_idx_to_bucket_idx_and_item_idx


if __name__ == "__main__":
    found_subclips = False
    with tqdm(total=200000, smoothing=0.0) as pbar:
        for filepath in video_data_table_root.rglob("*.mp4"):
            if "_" in filepath.parts[-1]:
                if not found_subclips:
                    found_subclips = True
                    pbar.update(1)
                continue
            else:
                found_subclips = False

            try:
                bucket_id, sample_id, video_id = filepath.parts[-4:-1]
                video_data = extract_frames_from_video(video_path=filepath)
                video_path_table_filepath = (
                    youtube_titles_clip_scores / bucket_id / sample_id
                )

                video_path_table = (
                    ds.dataset(video_path_table_filepath).to_table().to_pydict()
                )

                for idx, table_video_id in enumerate(
                    video_path_table["sorted_query_ids"]
                ):
                    if video_id in table_video_id:
                        reference_text = video_path_table["reference_text"][idx]
                        wit_idx = video_path_table["wit_idx"][idx]
                        term_idx = video_path_table["term_idx"][idx]
                        break

                clip_scores = get_clip_scores(
                    video_frames=video_data.unbind(0),
                    reference_text=reference_text,
                )
                relative_filepath = filepath.relative_to(video_data_table_root)
                add_row_to_table(
                    wit_idx=wit_idx,
                    term_idx=term_idx,
                    video_id=video_id,
                    filepath=relative_filepath,
                    reference_text=reference_text,
                    scores_sorted_idx=clip_scores.sorted_scores_idx.cpu()
                    .numpy()
                    .tolist(),
                    scores_sorted=clip_scores.scores_sorted.cpu().numpy().tolist(),
                )

                for starting_second_idx in clip_scores.sorted_scores_idx[
                    : args.num_subclips
                ]:
                    starting_second = floor(
                        starting_second_idx * args.subclip_length_in_seconds
                    )
                    output_path = filepath.parent / f"360p_{starting_second}.mp4"
                    trim_video(
                        video_path=filepath,
                        start_sec=starting_second,
                        end_sec=starting_second + args.subclip_length_in_seconds,
                        output_path=output_path,
                    )

            except Exception as e:
                logger.exception(e)
                pbar.update(1)
                continue

            pbar.update(1)
            filepath.unlink(missing_ok=True)
            pbar.set_description(f"Processed {filepath}")
