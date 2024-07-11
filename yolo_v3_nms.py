import torch
from yolo_v3_metrics import YOLOv3Metrics, anchor_box_list

# non_max_suppression과정은 YOLOv3Metrics의 함수를 재사용하자.
class Yolov3NMS(YOLOv3Metrics):
    def __init__(self, B=3, C=80, iou_th=0.4, conf_th=0.5):
        super().__init__(B, C, iou_th=iou_th, anchor=anchor_box_list)
        self.conf_th = conf_th #NMS의 첫번째 필터링 -> OS를 필터링하는 인자값

    # YOLOv3Metrics의 2개 함수 get_pred_bbox, compute_iou를 재사용한다.
    # get_pred_bbox는 재정의 해야함 -> 디바이스 문제...
    def get_pred_bbox(self, t_bbox, anchor, grid_x, grid_y, S):
        # 모든 텐서를 동일한 디바이스로 이동
        device = t_bbox.device
        anchor = anchor.to(device)
        grid_x = grid_x.to(device)
        grid_y = grid_y.to(device)

        # 여기에 들어오는 [tx], [ty], [tw], [th]는 모두 [Batch_size, S, S, B] 형태
        tx, ty = t_bbox[..., 0], t_bbox[..., 1]
        tw, th = t_bbox[..., 2], t_bbox[..., 3]

        # cnt는 작은 -> 중간 -> 큰 객체의 리스트를 순환하는 변수
        # anchor box list에서 cnt 값에 맞춰 객체 크기용 box [3x2]를 가져옴
        pw, ph = anchor[self.cnt][..., 0], anchor[self.cnt][..., 1]

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
    
    def compute_iou(self, box1, box2):
        return super().compute_iou(box1, box2)
    
    # NMS를 수행하는 준비과정인 데이터 추출 및 필터링 함수
    def process_predictions(self, pred):
        # 데이터 차원 배치 재정렬 하기
        pred = pred.permute(0, 2, 3, 1)
        assert pred.size(1) == pred.size(2), "데이터 정렬 오류"
            
        bs = pred.size(0)
        S = pred.size(1)

        # [batch_size, S, S, B*(C+5)]-> [batch_size, S, S, B, (C+5)]
        prediction = pred.view(bs, S, S, self.B, -1)

        # tx, ty, OS(objectness score), CP(class probability 항목에 sigmoid 적용
        prediction[..., :2] = torch.sigmoid(prediction[..., :2])  # t_x, t_y
        prediction[..., 4:] = torch.sigmoid(prediction[..., 4:])  # OS, CP

        t_bboxes = prediction[..., :4] #[batch_size, S, S, B, 4]
        obj_score = prediction[..., 4] #[batch_size, S, S, B]
        cp_score = prediction[..., 5:] #[batch_size, S, S, B, C]

        # 그리드셀 좌표 색인을 위한 매쉬그리드 데이터 생성
        grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')

        # [tx, ty, tw, th] -> [bx, by, bw, bh] 변환 실행
        b_bboxes = self.get_pred_bbox(t_bboxes, self.anchor, grid_x, grid_y, S)

        # OS 정보를 conf_th로 필터링 (1차)
        # 각각 마스크 필터를 통과하여 [N, ~~] 차원으로 변환된다.
        conf_mask = obj_score > self.conf_th
        b_bboxes = b_bboxes[conf_mask] # [N, 4]
        obj_score = obj_score[conf_mask] # [N]
        cp_score = cp_score[conf_mask] # [N, C]

        # class confidence 곱하기
        # 이때 obj_score는 broadcasting이 안되기에 맨 뒤에 차원 추가
        cp_score *= obj_score.unsqueeze(-1)

        # 클래스와 스코어 정보로 필터링(2차)
        cp_val, cp_idx = torch.max(cp_score, -1)
        conf_mask = cp_val > self.conf_th
        b_bboxes = b_bboxes[conf_mask]
        cp_val = cp_val[conf_mask]
        cp_idx = cp_idx[conf_mask]

        return b_bboxes, cp_val, cp_idx
    
    # 필터링 + 내림차순을 거친 데이터를 기반으로 의미있는 bbox만 선택
    def apply_nms(self, b_bboxes, cp_val, cp_idx):
        keep_boxes = []
        while b_bboxes.size(0):
            # 신뢰도값이 가장 큰 단 1개의 데이터만 추출
            chosen_box = b_bboxes[0].unsqueeze(0) #[1, 4]
            chosen_val = cp_val[0].unsqueeze(0).unsqueeze(0) #[1, 1]
            chosen_idx = cp_idx[0].unsqueeze(0).unsqueeze(0) #[1, 1]

            keep_boxes.append(torch.cat((chosen_box, chosen_val, chosen_idx), dim=-1))
            if b_bboxes.size(0) == 1:
                break
            
            # 신뢰도값이 가장 큰 box랑 나머지 bbox를 비교하여 iou 연산
            # 연산하여 iou가 겹치는 bbox 항목을 제거한다.
            ious = self.compute_iou(chosen_box, b_bboxes[1:])
            b_bboxes = b_bboxes[1:][ious < self.iou_th]
            cp_val = cp_val[1:][ious < self.iou_th]
            cp_idx = cp_idx[1:][ious < self.iou_th]
        return keep_boxes


    def non_max_suppression(self, outputs):
        f_b_bboxes = []
        f_cp_val = []
        f_cp_idx = []

        # outputs 리스트 정보를 원소로 분해
        for cnt, pred in enumerate(outputs):
            self.cnt = cnt # 리스트 순회값으로 cnt 초기화
            
            b_bboxes, cp_val, cp_idx = self.process_predictions(pred)
            f_b_bboxes.append(b_bboxes)
            f_cp_val.append(cp_val)
            f_cp_idx.append(cp_idx)

        # 리스트 원소를 concat하기
        nms_bboxes = torch.cat(f_b_bboxes, dim=0)
        nms_val = torch.cat(f_cp_val, dim=0)
        nms_idx = torch.cat(f_cp_idx, dim=0)

        # 내림차순 정렬(신뢰도가 큰 값이 맨 앞에 오게)
        sorted_indices = torch.argsort(-nms_val)
        nms_bboxes = nms_bboxes[sorted_indices]
        nms_val = nms_val[sorted_indices]
        nms_idx = nms_idx[sorted_indices]

        # NMS 적용
        res_boxes = self.apply_nms(nms_bboxes, nms_val, nms_idx)
        # 최종 출력되는 res boxes는
        # 1) 신뢰도 값이 가장 높은 boxes 정보이다.
        # 2) 그리고 서로 겹치지 않는다(iou로 필터링됨)
        
        # 마지막에 출력되는 res_boxes의 원소값은 모두 [1,6]형태이니 [6]으로 차원축소
        # [bbox좌표, 신뢰도값, 클래스 id] 순이다.
        res_boxes = [box.squeeze(0) for box in res_boxes]

        return res_boxes
    

# 디버깅을 위한 nms_debug 함수
def nms_debug():
    S_list = [52, 26, 13]
    outputs = []

    for S in S_list:
        output = torch.rand(1, 255, S, S)
        outputs.append(output)

    nms = Yolov3NMS()
    boxes = nms.non_max_suppression(outputs)

    for box in boxes:
        print(box)


if __name__ == '__main__':
    nms_debug()