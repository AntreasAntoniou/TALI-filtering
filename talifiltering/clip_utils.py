from dataclasses import dataclass
from typing import List, Union
import numpy as np

import torch
import tqdm
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

model.to(torch.cuda.current_device())
model.eval()


@dataclass
class CLIPScoreOutput:
    scores: List[float]
    sorted_scores_idx: List[str]
    reference_text: str


def get_scores(
    reference_text: str,
    image_list: List[torch.Tensor],
) -> CLIPScoreOutput:

    reference_text_inputs = processor(
        text=reference_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    reference_text_inputs = reference_text_inputs.to(torch.cuda.current_device())
    image_inputs = processor(
        images=image_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    image_inputs.to(torch.cuda.current_device())

    with torch.no_grad():
        reference_text_features = model.get_text_features(**reference_text_inputs)
        image_features = model.get_image_features(**image_inputs)
        reference_text_features = (
            reference_text_features
            / reference_text_features.norm(p=2, dim=-1, keepdim=True)
        )
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    scores = reference_text_features @ image_features.T

    sorted_scores_idx = torch.argsort(scores[0], dim=-1, descending=True)

    return CLIPScoreOutput(
        sorted_scores_idx=sorted_scores_idx.detach().cpu(),
        scores=scores.detach().cpu(),
        reference_text=reference_text,
    )


if __name__ == "__main__":
    from rich import print
    from rich.traceback import install

    install()

    print("Model being used: ", model)
    print("Model parameters")

    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad, param.dtype)
    with tqdm.tqdm(total=10000) as pbar:
        reference_text = "A cute dog"

        for _ in range(25000):
            for batch_size in [800]:
                image_tensor = torch.randn(
                    (batch_size, 3, 224, 224), dtype=torch.float32
                )
                image_list = image_tensor.unbind(0)

                output = get_scores(
                    reference_text=reference_text, image_list=image_list
                )

                # print(batch_size, output.scores, output.scores.dtype)
                # print(output.sorted_scores_idx)
                pbar.update(1)
