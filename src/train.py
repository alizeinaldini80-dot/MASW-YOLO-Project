
#src/train.py

import argparse
import os

import yaml
from ultralytics import YOLO

from . import models  # noqa: F401

from .utils.nms import enable_soft_nms

from .utils.wiou import enable_wiou

def load_config(exp_config_path):
    with open('configs/base_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    with open(exp_config_path, 'r') as f:
        exp_cfg = yaml.safe_load(f)
    cfg.update(exp_cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    project_path = cfg['project']
    if project_path.startswith('/content/drive') and not os.path.isdir('/content/drive/MyDrive'):
        raise RuntimeError("❌")
    os.makedirs(project_path, exist_ok=True)

    if cfg.get('soft_nms', False):
        enable_soft_nms(
            method=cfg.get('soft_nms_method', 'gaussian'),
            sigma=cfg.get('soft_nms_sigma', 0.5),
            score_thres=cfg.get('soft_nms_score_thres', 0.001),
        )

    if cfg.get('wiou', False):
        enable_wiou(
            alpha=cfg.get('wiou_alpha', 1.9),
            delta=cfg.get('wiou_delta', 3.0),
            momentum=cfg.get('wiou_momentum', 0.9999),
        )
    checkpoint_path = os.path.join(cfg['project'], cfg['name'], 'weights', 'last.pt')
    resume = os.path.exists(checkpoint_path)

    if resume:
       
        model = YOLO(checkpoint_path)
    elif 'model_yaml' in cfg:
        model = YOLO(cfg['model_yaml'])
        model.load(cfg['weights'])
    else:
        model = YOLO(cfg['weights'])
    if resume:
        results = model.train(resume=True)
    else:
        results = model.train(
            data=cfg['data'],
            epochs=cfg['epochs'],
            batch=cfg['batch'],
            imgsz=cfg['imgsz'],
            optimizer=cfg.get('optimizer', 'SGD'),
            lr0=cfg['lr0'],
            lrf=cfg['lrf'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            device=cfg['device'],
            workers=cfg['workers'],
            project=cfg['project'],
            name=cfg['name'],
            exist_ok=cfg['exist_ok'],
            verbose=cfg['verbose'],
        )


if __name__ == "__main__":
    main()
