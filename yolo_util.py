import torch
import coco_data

class YoloComFunc:
    def __init__(self, anchor=coco_data.anchor_box_list,
                 S_list=coco_data.S_list):
        self.S_list = S_list
        self.anchor = anchor

    def get_pred_bbox(self, t_bbox):
        device = t_bbox.device
        S = t_bbox.size(1)
        self.anchor = self.anchor.to(device)

        # 새로이 변수를 생성하는 매쉬그리드 변수는 디바이스가 어디인지 알려줘야함
        grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
        grid_y = grid_y.to(device)
        grid_x = grid_x.to(device)

        # 여기에 들어오는 [tx], [ty], [tw], [th]는 모두 [Batch_size, S, S, B] 형태
        tx, ty = t_bbox[..., 0], t_bbox[..., 1]
        tw, th = t_bbox[..., 2], t_bbox[..., 3]

        anchor_idx = self.S_list.index(S)

        # anchor box list에서 cnt 값에 맞춰 객체 크기용 box [3x2]를 가져옴
        pw = self.anchor[anchor_idx][..., 0]
        ph = self.anchor[anchor_idx][..., 1]

        # meshgrid로 정의된 grid_x, grid_x는 모두 [S, S] 차원을 가짐
        # [S, S] -> [1, S, S, 1] -> [batch_size, S, S, B]로 차원 확장
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand_as(tx)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand_as(ty)

        # grid / S -> 그리드셀의 좌상단 좌표값이 됨(cx, cy)
        # tx, ty는 sigmoid 처리된 c_pos와 b_pos의 '상대적인 거리'
        # 모두 정규화 좌표 평면이니 t_pos도 1/S 처리 해줘야 함
        bx = (tx + grid_x) / S
        by = (ty + grid_y) / S

        bw = pw * torch.exp(tw)
        bh = ph * torch.exp(th)
        
        # 좌표 변환이 완료된 [bx, by, bw, bh]은 스택으로 쌓아서 return
        b_bbox = torch.stack([bx, by, bw, bh], dim=-1)
        return b_bbox
    

        # iou계산 함수는 뭐 딱히 어려울 건 없다.
    def compute_iou(self, box1, box2):
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2

        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2

        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        union_area = box1_area + box2_area - inter_area

        # 박스가 겹치지 않는 경우 -> unioun_area가 0이 됨 -> 이때 NaN발생 가능성 존재
        ious = torch.where(union_area > 0, inter_area / union_area, 
                                           torch.zeros_like(union_area))
        return ious