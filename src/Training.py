model_yaml_path = r"E:/FYWL/src/yolo11.yaml"
data_yaml_path = r"E:/FYWL/datal/datal.yaml"
pre_yaml_name = r"E:/FYWL/yolo11s.pt"

from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO(model_yaml_path)
    model.load('yolo11s.pt')
    model.train(data=data_yaml_path,
                imgsz = 640,
                epochs = 50,
                batch = 16,
                workers = 1,
                optimizer = 'SGD',
                amp = True,
                verbose = True,
                )
