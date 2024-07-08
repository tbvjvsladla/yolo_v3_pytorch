import os
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image


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

class CustomDataset(Dataset):
    def __init__(self, root, img_set=None, transform=None, 
                 S=[52, 26, 13], B=3, C=80,
                 anchor = None):
        self.root = root #COCO 데이터셋의 메인 폴더 경로

        self.img_set = img_set
        self.transform = transform
        self.S_list = S # Gird Cell사이즈 -> [13, 26, 52] -> 인자값으로 받기
        self.B = B # anchorbox의 개수 : 3
        self.C = C # COCO데이터셋의 클래스 개수 : 80

        # Anchor_box_list를 관리하기 위한 변수
        self.anchor = anchor

        self.anno_path = os.path.join(root, 'annotations')

        if img_set == 'train':
            self.img_dir = os.path.join(root, 'my_train_img')
            self.coco = COCO(os.path.join(self.anno_path, 'instances_val2014.json'))
        elif img_set == 'val':
            self.img_dir = os.path.join(root, 'my_val_img')
            self.coco = COCO(os.path.join(self.anno_path, 'instances_val2017.json'))
        elif img_set == 'test':
            pass
        else:
            raise ValueError("이미지 모드 확인하기")
        
        # 모드별 이미지 ID리스트 불러오기
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)
    
    #예의 바르게 해당 클래스의 실행 결과를 불러오자
    def __str__(self):
        if len(self.img_ids) == 0:
            message = f"Img 폴더 내 파일 못찾음"
        else:
            message = f"Img 폴더 찾은 *.jpg 개수: {len(self.img_ids)}"
        return f"{message}"
    
    # label matrix의 [b*x, b*y, b*w, b*h]를 [t*x, t*y, t*w, t*h]로 변환하는 코드
    def _reverse_gt_bbox(self, b_bbox, anchor, cell, S):
        if self.anchor is None:
            raise ValueError("앵커박스 정보가 없음")
        
        # [b*x, b*y, b*w, b*h]는 0~1범위로 스케일링 되었으니
        # 시그모이드 역함수를 적용하지 않아도 된다.
        bx, by, bw, bh = b_bbox
        pw, ph = anchor
        grid_y, grid_x = cell

        tx = bx * S - grid_x
        ty = by * S - grid_y

        tw = torch.log(bw / pw)
        th = torch.log(bh / ph)

        t_bbox = [tx, ty, tw, th]
        return t_bbox
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # ann_ids는 불러온 이미지와 관련된 Anno 리스트를 가져온다
        anns = self.coco.loadAnns(ann_ids)

        # 이미지의 기본 정보 추출하기 -> [0]붙이는거 까먹지말자
        img_info = self.coco.loadImgs(img_id)[0]
        # 찾은 이미지에 맞는 파일이름을 찾아서 출력할 Image에 할당
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # 라벨 메트릭스 생성 [S, S, C+5=85]로 생성을 먼저함
        label_matrix = [
            torch.zeros((S, S, self.C+5)) for S in self.S_list
        ]
        # 이제 Anno주석정보를 라벨 매트릭스에 기입하기
        for ann in anns:
            bbox = ann['bbox']
            # CP는 Class Probability이니 = Class ID
            CP_idx = real_class_idx[ann['category_id']]

            x, y, w, h = bbox
            # bbox 좌표 정규화
            bx = (x + w / 2) / img_info['width']
            by = (y + h / 2) / img_info['height']
            bw = w / img_info['width']
            bh = h / img_info['height']

            #정규좌표평면에서 center point가 위치하는 그리드셀 탐색
            for i, S in enumerate(self.S_list):
                cell_size_x = img_info['width'] / S
                cell_size_y = img_info['height'] / S
                grid_x = int(bx / cell_size_x)
                grid_y = int(by / cell_size_y)

                #탐색한 그리드셀이 비어있으면 아래 정보 기입
                if label_matrix[i][grid_y, grid_x, 4] == 0:
                    # OS(Objectness Score) 정보 기입
                    label_matrix[i][grid_y, grid_x, 4] = 1
                    # BBox(bx, by, bw, bh) 정보 기입
                    label_matrix[i][grid_y, grid_x, :4] = torch.tensor(
                        [bx, by, bw, bh]
                    )
                    # CP(Class Probability) 정보 기입
                    label_matrix[i][grid_y, grid_x, 5 + CP_idx] = 1

        # 이미지의 전처리 방법론이 수행될 때 label matrix도 같이 업데이트
        if self.transform:
            image = self.transform(image)

            for i, S in enumerate(self.S_list):
                bxby = label_matrix[i][..., :4]
                # 정보가 있는 matrix탐색 -> 마스크 필터링을 활용
                mask = label_matrix[i][..., 4] == 1
                bxby = bxby[mask]
                # 정보가 있는 그리드셀의 idx를 다시 서치
                grid_cells = torch.nonzero(mask, as_tuple=False)

                # repeat를 사용하여 [S, S, C+5] -> [S, S, B*(C+5)]
                label_matrix[i] = label_matrix[i].repeat(1, 1, self.B)

                for b in range(self.B):
                    anchor = self.anchor[i][b]
                    # 마스크 필터를 통과한 정보가 있는 데이터만 찾아야함
                    for j in range(len(bxby)):
                        cell = grid_cells[j]
                        t_bbox = self._reverse_gt_bbox(bxby[j], anchor, cell, S)
                        # 계산된 t_bbox정보를 해당 셀에 맞춰 치환
                        idx = (self.C+5) * b
                        label_matrix[i][cell[0], cell[1], idx:idx+4] = torch.tensor(t_bbox)

        return image, tuple(label_matrix)
    

if __name__ == '__main__':
    pass