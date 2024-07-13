import torch


# 슈퍼 클래스와 해당 색상 매핑
cls_color = {
    "person": "magenta",
    "vehicle": "yellow",
    "outdoor": "blue",
    "animal": "green",
    "accessory": "purple",
    "sports": "orange",
    "kitchen": "brown",
    "food": "red",
    "furniture": "cyan",
    "electronic": "pink",
    "appliance": "lime",
    "indoor": "gray"
}

# COCO 클래스 레이블을 슈퍼 클래스에 따라 그룹화한 딕셔너리
coco_map = {
    "person": [0],
    "vehicle": [1, 2, 3, 4, 5, 6, 7],
    "outdoor": [8, 9, 10, 11, 12, 13],
    "animal": [14, 15, 16, 17, 18, 19, 20, 21, 22],
    "accessory": [23, 24, 25, 26, 27],
    "sports": [28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
    "kitchen": [38, 39, 40, 41, 42, 43, 44],
    "food": [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    "furniture": [56, 57, 58],
    "electronic": [59, 60, 61, 62, 63, 64],
    "appliance": [65, 66, 67, 68],
    "indoor": [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
}

# COCO 클래스 레이블
coco_label = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 클래스 ID를 슈퍼 클래스 이름으로 매핑
cls_map = {cls_id: superclass for superclass, cls_ids in coco_map.items() for cls_id in cls_ids}

# 사용되지 않는 category_id를 제외한 딕셔너리 생성
used_categories = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 
    89, 90
]

real_class_idx = {cat_id: idx for idx, cat_id in enumerate(used_categories)}


# K-mean Clustering을 수행하여 얻은 9개의 anchorbox
# 해당 데이터는 정규화 좌표평면에서 얻은 Anchorbox 크기값이다.
anchor_box_list = torch.tensor([[[0.0686, 0.0861],
                                 [0.1840, 0.2270],
                                 [0.2308, 0.4469]],

                                [[0.4641, 0.2762],
                                 [0.3029, 0.7475],
                                 [0.5187, 0.5441]],

                                [[0.8494, 0.4666],
                                 [0.5999, 0.8385],
                                 [0.9143, 0.8731]]])
