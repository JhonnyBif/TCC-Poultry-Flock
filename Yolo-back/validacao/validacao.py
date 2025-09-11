from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
GITHUB_ASSETS_STEMS
from ultralytics import YOLO, settings
settings

if __name__ == '__main__':
  settings.update({'weights_dir': './weights'})
  settings.update({'datasets_dir': './datasets'})
  settings.update({'runs_dir': './runs'})

  model = YOLO('runs/detect/train6/weights/best.pt')
  results = model.val(
                      imgsz=640,
                      batch=16,
                      conf=0.5,
                      iou=0.7,  # Non-Maximum Supression (NMS)
                      save_json=False,  # Save to JSON {image_id, cls, bbox, conf} of each image in dataset
                      save_hybrid=False,  # Bounding box labels + inference on the output image (Ultralytics 8.0.178 = results are wrong when this argmunent is True)
                      split='test')  # train, val or test
