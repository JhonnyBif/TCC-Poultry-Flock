from ultralytics.utils.torch_utils import strip_optimizer

strip_optimizer('runs/detect/train4/weights/best.pt', 'runs/detect/train4/weights/best.pt')