# from ultralytics import YOLO
# if __name__ == '__main__':
#     # Load a model
#     model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

#     # Train the model
#     results = model.train(data="D:/projects/yolov11/from_data_2024_12_purple_only_v2(70%)_balance", epochs=100, imgsz=640, device ='cuda', batch = 1, workers=1)
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n-cls.pt")  # 加载预训练模型

    training_params = {
        "data": "D:/projects/yolov11/from_data_2024_12_purple_only_v2(70%)",
        "epochs": 100,
        "imgsz": 640,
        "batch": 4,
        "device": 'cuda',
        "workers": 1,
        "augment": True,  # 启用数据增强
        "degrees": 100.0,  # 旋转角度范围
        "translate": 0.05,  # 平移比例
        "scale": 1.0,  # 缩放比例
        "shear": 0.0,  # 剪切角度
        "perspective": 0.0,  # 透视增强
        "hsv_h": 0.015,  # 色调增强
        "hsv_s": 0.5,  # 饱和度增强
        "hsv_v": 0.5,  # 明度增强
        "flipud": 0.5,  # 垂直翻转概率
        "fliplr": 0.5,  # 水平翻转概率
        "mosaic": 0.0,  # Mosaic 数据增强概率
        "mixup": 0.0,  # Mixup 数据增强概率
        "copy_paste": 0.0,
        "auto_augment": 'randaugment',
        "erasing": 0.0,
        "crop_fraction": 1.0
    }

    results = model.train(**training_params)