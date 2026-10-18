"""Load workflow detectors, optionally using a few recorded visual exemplars."""
import os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).parent/'Ultralytics'))
import numpy as np
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

LABELS = ['gloved hand', 'bottle', 'multiwell plate', 'pipette', 'pipette controller', 'spray bottle']
REFERENCE = Path('output/real_workflow/probe/video-108_110.jpg')
# One visual exemplar per first three classes. These are prompt boxes, not a training set.
REFERENCE_BOXES = [[173, 657, 371, 954], [177, 491, 289, 716], [337, 726, 523, 869]]


def load_detector(mode='text', image_size=640):
    model = YOLOE('yoloe-26s-seg.pt')
    if mode == 'visual':
        cached = Path('output/real_workflow/visual_prompts.npz')
        if cached.exists():
            model.load_prompt_embeddings(cached)
        else:
            # Text encoder is invoked before switching the detector to FP16.
            text_pe = model.get_text_pe(LABELS[3:]).detach()
            model.predict(str(REFERENCE), refer_image=str(REFERENCE), visual_prompts={'bboxes':np.array(REFERENCE_BOXES), 'cls':np.arange(3)}, predictor=YOLOEVPSegPredictor, device=0, imgsz=image_size, verbose=False)
            visual_pe = model.model.pe.detach()
            embeddings = torch.cat([visual_pe.float(),text_pe.to(visual_pe.device).float()],dim=1)
            model.set_classes(LABELS, embeddings)
            model.save_prompt_embeddings(cached)
    else:
        model.set_classes(LABELS)
    return model
