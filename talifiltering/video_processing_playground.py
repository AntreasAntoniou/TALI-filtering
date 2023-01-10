from math import floor
import torch
import torchvision
from torchvision.datasets.utils import download_url
import torch
import json
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
)
from typing import Dict

from rich import print
from rich.traceback import install

install()


# Load pre-trained model
model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)

download_url(
    "https://github.com/pytorch/vision/blob/main/test/assets/videos/WUzgd7C1pWA.mp4?raw=true",
    ".",
    "WUzgd7C1pWA.mp4",
)
video_path = "./WUzgd7C1pWA.mp4"

# Load video
video: EncodedVideo = EncodedVideo.from_path(video_path)
clip_start_sec = 0.0  # secs
video_duration_seconds = float(video.duration)

# Get clip
print(f"Video duration: {video_duration_seconds} seconds")
video_data = video.get_clip(
    start_sec=clip_start_sec, end_sec=clip_start_sec + video_duration_seconds
)

audio_sample_rate = floor(video_data["audio"].shape[0] / video_duration_seconds)
video_sample_rate = floor(video_data["video"].shape[1] / video_duration_seconds)
num_frames = video_duration_seconds


video_data["video"] = video_data["video"][
    :, : video_sample_rate * floor(video_duration_seconds)
]


video_data["audio"] = video_data["audio"][
    : audio_sample_rate * floor(video_duration_seconds)
]
video_data["audio"] = video_data["audio"].view(
    floor(video_duration_seconds), audio_sample_rate
)

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

audio_transform = ApplyTransformToKey(
    key="audio",
    transform=Compose(
        [
            UniformTemporalSubsample(num_samples=floor(num_frames), temporal_dim=0),
        ]
    ),
)


video_data = video_transform(video_data)
audio_data = audio_transform(video_data)

for key, value in video_data.items():
    print(f"{key}: {value.shape}")
