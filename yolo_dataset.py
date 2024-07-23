import os
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
import coco_data #데이터 관리용 py파일

class CustomDataset(Dataset):
    def __init__(self, root, load_anno=None, transform=None, 
                 S=coco_data.S_list, B=3, C=80,
                 anchor = None):
        self.root = root #COCO 데이터셋의 메인 폴더 경로

        self.load_anno = load_anno
        self.transform = transform
        self.S_list = S # Gird Cell사이즈 -> [13, 26, 52] -> 인자값으로 받기
        self.B = B # anchorbox의 개수 : 3
        self.C = C # COCO데이터셋의 클래스 개수 : 80

        # Anchor_box_list를 관리하기 위한 변수
        self.anchor = anchor

        self.anno_path = os.path.join(root, 'annotations')

        if load_anno is None:
            raise ValueError("주석파일 정보기입")
        else:
            self.img_dir = os.path.join(root, load_anno)
            anno_file = 'instances_' + load_anno + '.json'
            self.coco = COCO(os.path.join(self.anno_path, anno_file))
        
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
    def _reverse_gt_bbox(self, b_bbox, anchor, cell_idx, S):
        if self.anchor is None:
            raise ValueError("앵커박스 정보가 없음")
        
        # [b*x, b*y, b*w, b*h]는 0~1범위로 스케일링 되었으니
        # 시그모이드 역함수를 적용하지 않아도 된다.
        bx, by, bw, bh = b_bbox
        pw, ph = anchor
        # 정규화 좌표평면에서 그리드셀의 좌상단 Pos정보 생성
        cy, cx = cell_idx / S

        # 정규 좌표평면 상에서 cx, cy와  bx, by의
        # 상대적인 거리가 sigmoid(tx), sigmoid(ty)이다.
        sig_tx = (bx - cx) * S
        sig_ty = (by - cy) * S

        # tw, th는 앵커박스와 b_box간의 크기 비율정보를 의미함
        # 이때 bw, bh, pw, ph가 [0~1] 정규좌표평면상의 데이터여서
        # 역함수 t = ln(b/p)를 수행 시 NaN이 발생할 수 있다.
        # 따라서 스케일링 + 0 이 되는것 방지 두개의 안전장치를 걸어둔다.
        scale_factor = 1000
        eclipse = 1e-6
        bw_scaled = bw * scale_factor + eclipse
        bh_scaled = bh * scale_factor + eclipse
        pw_scaled = pw * scale_factor + eclipse
        ph_scaled = ph * scale_factor + eclipse
        tw = torch.log(bw_scaled / pw_scaled)
        th = torch.log(bh_scaled / ph_scaled)

        t_bbox = [sig_tx, sig_ty, tw, th]
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
            CP_idx = coco_data.real_class_idx[ann['category_id']]

            x, y, w, h = bbox
            # bbox 좌표 정규화
            bx = (x + w / 2) / img_info['width']
            by = (y + h / 2) / img_info['height']
            bw = w / img_info['width']
            bh = h / img_info['height']

            #정규좌표평면에서 center point가 위치하는 그리드셀 탐색
            for i, S in enumerate(self.S_list):
                # 정규화된 좌표평면 상에서 grid_y, x의 idx를 구해야 한다.
                grid_x = int(bx * S)
                grid_y = int(by * S)

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
                b_boxes = label_matrix[i][..., :4]
                # 정보가 있는 matrix탐색 -> 마스크 필터링을 활용
                mask = label_matrix[i][..., 4] == 1
                b_boxes = b_boxes[mask]
                # 정보가 있는 그리드셀의 idx를 다시 서치
                grid_cells = torch.nonzero(mask, as_tuple=False)

                # repeat를 사용하여 [S, S, C+5] -> [S, S, B*(C+5)]
                label_matrix[i] = label_matrix[i].repeat(1, 1, self.B)

                for b in range(self.B):
                    anchor = self.anchor[i][b]
                    # 마스크 필터를 통과한 정보가 있는 데이터만 찾아야함
                    for j in range(len(b_boxes)):
                        cell_idx = grid_cells[j]
                        t_bbox = self._reverse_gt_bbox(b_boxes[j], anchor, cell_idx, S)
                        # 계산된 t_bbox정보를 해당 셀에 맞춰 치환
                        idx = (self.C+5) * b
                        label_matrix[i][cell_idx[0], cell_idx[1], idx:idx+4] = torch.tensor(t_bbox)

        return image, tuple(label_matrix)
    
    # 디버그용 img 정보 취득 함수
    def get_img_info(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        img_info['obj_num'] = len(ann_ids)
        return img_info
    

if __name__ == '__main__':
    pass