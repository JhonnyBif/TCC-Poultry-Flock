from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
GITHUB_ASSETS_STEMS
from ultralytics import YOLO, settings
settings

if __name__ == '__main__':
  settings.update({'weights_dir': './weights'})
  settings.update({'datasets_dir': './datasets'})
  settings.update({'runs_dir': './runs'})

  model = YOLO('yolo11m')

              # model='./runs/detect/train/weights/last.pt',
  model.train(data='yamls/custom_dataset-chickenDetection-video.yaml',
              epochs=400,
              patience=80,  # early stopping patience (0 to disable)
              batch=.8,  # number of images per batch (-1 for AutoBatch)
              imgsz=640,
              workers=8,
              pretrained=True,
              resume=False,  # resume training from last checkpoint
              single_cls=False,  # Whether all classes will be the same (just one class)
              box=6.5,  # More recall, better IoU, less precission, 
              cls=0.7,  # Bbox class better
              dfl=1.5,  # Distribution Focal Loss. Better bbox boundaries
              val=True,
              # Augmentations
              degrees=0.3,
              hsv_s=0.4,
              hsv_v=0.4,
              scale=0.3,
              fliplr=0.5
            )
