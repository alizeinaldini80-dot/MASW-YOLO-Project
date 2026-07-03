"""
src/train.py

اسکریپت یکپارچه آموزش برای همه آزمایش‌های پروژه MASW-YOLO.

نحوه اجرا (حتماً از ریشه پروژه، نه از داخل src/):
    python -m src.train --config configs/exp1_msca.yaml
"""

import argparse
import yaml
from ultralytics import YOLO

# import نسبی: این خط باعث اجرای src/models/__init__.py می‌شود
# که ماژول‌های سفارشی (MSCA) را در Ultralytics ثبت می‌کند
from . import models  # noqa: F401


def load_config(exp_config_path):
    with open('configs/base_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    with open(exp_config_path, 'r') as f:
        exp_cfg = yaml.safe_load(f)
    cfg.update(exp_cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                         help='مسیر فایل کانفیگ آزمایش، مثال: configs/exp1_msca.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if 'model_yaml' in cfg:
        model = YOLO(cfg['model_yaml'])
        model.load(cfg['weights'])
        print(f"مدل با معماری سفارشی ساخته شد: {cfg['model_yaml']}")
    else:
        model = YOLO(cfg['weights'])
        print(f"مدل مستقیم از وزن‌های آماده بارگذاری شد: {cfg['weights']}")

    results = model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        batch=cfg['batch'],
        imgsz=cfg['imgsz'],
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

    print(f"✅ آموزش '{cfg['name']}' با موفقیت به پایان رسید!")
    print(f"وزن‌های نهایی: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
