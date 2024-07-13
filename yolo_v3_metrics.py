import torch
import torch.nn as nn
import torch.nn.functional as F


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


class YOLOv3Metrics:
    def __init__(self, B=3, C=80, iou_th = 0.5,
                 anchor=None, device='cuda'):
        self.B = B
        self.C = C
        self.device = device # 연산에 필요한 변수를 다 GPU로 올려야 함

        self.iou_th = iou_th
        # anchor 리스트 정보는 GPU로 올려야 함
        self.anchor = anchor.to(device) if anchor is not None else None

    def cal_acc_func(self, outputs, labels):
        # outputs, labels가 리스트 데이터 이기에
        # 원소별로 4가지 평가지표를 계산 후 덧셈한다.
        sum_iou = 0
        sum_precision = 0
        sum_recall = 0
        sum_top1_error = 0

        # outputs와 labels를 GPU로 이전 + output의 데이터 정렬 수행
        self.outputs = [output.to(self.device).permute(0, 2, 3, 1) for output in outputs]
        self.labels = [label.to(self.device) for label in labels]

        # 리스트 내 원소가 0, 1, 2 3개니까 cnt는 0~2 를 순회한다.
        self.cnt = 0 # compute_metrics 함수 수행 전 cnt 초기화

        # 리스트 내 원소들을 추출한 후 연산 수행
        for output, label in zip(self.outputs, self.labels):
            # 실제 원소별 평가지표 연산은 `compute_metrics`함수에서 수행
            element_metrics = self.compute_metrics(output, label)
            
            sum_iou += element_metrics[0]
            sum_precision += element_metrics[1]
            sum_recall += element_metrics[2]
            sum_top1_error += element_metrics[3]

            self.cnt += 1

        iou = sum_iou / self.cnt if self.cnt > 0 else 0
        precision = sum_precision / self.cnt if self.cnt > 0 else 0
        recall = sum_recall / self.cnt if self.cnt > 0 else 0
        top1_error = sum_top1_error / self.cnt if self.cnt > 0 else 0

        # YOLOv3Metrics클래스의 최종 리턴값이라 보면 된다.
        cal_res = [iou, precision, recall, top1_error]
        return cal_res
    

    def compute_metrics(self, output, label):
        # Assert 구문을 활용하여 grid cell의 크기가 동일한지 확인
        assert output.size(1) == label.size(2), "그리드셀 차원이 맞지 않음"

        batch_size = output.size(0) # Batch_size의 사이즈정보
        S = output.size(2) # grid cell크기정보 추출

        # 원소의 차원은 [batch_size, S, S, Bx(5+C)] 4차원으로 되어있음
        # 이걸 [batch_size, S, S, B, 5+C] 5차원으로 차원전환
        predictions = output.view(batch_size, S, S, self.B, 5 + self.C)
        target = label.view(batch_size, S, S, self.B, 5 + self.C)

        # BBox정보 추출(tx, ty, tw, th) + OS정보도 포함시킴
        # 따라서 [batch_size, S, S, B, 5]
        pred_boxes = predictions[..., :5]
        target_boxes = target[..., :5]

        # BBox의 [tx, ty, tw, th] -> [bx, by, bw, bh]로 좌표변환
        b_pred_boxes, b_target_boxes = self.convert_b_bbox(pred_boxes, target_boxes)
        # 좌표변환된 b_pred_boxes, b_target_boxes의 차원은 
        # pred_boxes, target_boxes에서 OS정보만 빠짐 [batch_size, S, S, B, 4]

        # 1번 평가지표 = 원소 IOU연산하기
        # 입력차원 : [batch_size, S, S, B, 4]
        ious = self.compute_iou(b_pred_boxes, b_target_boxes)
        iou_score = ious.mean().item()

        # IOU 임계값을 이용하여 TP, FP, FN 계산
        pred_os = (ious > self.iou_th).float()
        target_os = (target_boxes[..., 4] > 0).float()
        TP = (pred_os * target_os).sum().item()
        FP = (pred_os * (1 - target_os)).sum().item()
        FN = ((1 - pred_os) * target_os).sum().item()

        # 2,3번 평가지표 원소 정밀도(precision), 원소 재현율(recall) 계산
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 클래스 확률 정보(Class Probability) 추출 후 차원전환
        # [Batch_size, S, S, C] -> [Batch_size*S*S, C]
        # 원래는 pred_cp에 sigmoid를 적용해야 하나 패스...
        pred_cp = predictions[..., 5:].reshape(-1, self.C)
        target_cp = target[..., 5:].reshape(-1, self.C)

        # Softmax 적용 -> pred만 적용하면 됨
        pred_cp = F.softmax(pred_cp, dim=1)

        # Class 확률 정보에서 가장 높은 값을 갖는 idx 정보 추출
        pred_classes = torch.argmax(pred_cp, dim=1)
        target_classes = torch.argmax(target_cp, dim=1)
        # 추출 결과물은 [Batch_size*S*S] 1차원이 된다.

        # 객체가 아에 검출되지 않는 예외현상을 처리하는 구문
        non_zero_mask = target_classes != 0
        f_pred_classes = pred_classes[non_zero_mask]
        f_target_classes = target_classes[non_zero_mask]

        # 4번 평가지표 원소 Top-1 Error 계산
        top1_error = (~f_pred_classes.eq(f_target_classes)).float().mean().item()

        #원소 평가지표 4가지를 리턴
        element_metrics = [iou_score, precision, recall, top1_error]
        return element_metrics
    
    # [tx, ty, tw, th] -> [bx, by, bw, bh]의 수행을 준비하는 함수
    def convert_b_bbox(self, pred_boxes, target_boxes):
        # 마스크 필터 생성하기(yolo v3 Loss와 동일한 방식
        target_obj = target_boxes[..., 4]
        coord_mask = target_obj > 0
        noobj_mask = target_obj <= 0
        #예외처리 구문
        if coord_mask.sum() == 0:
            b_pred_boxes = torch.zeros_like(pred_boxes[..., :4])
            b_target_boxes = torch.zeros_like(target_boxes[..., :4])
            return b_pred_boxes, b_target_boxes

        # 필터링의 응용 -> 의미없는 셀을 탈락시키지 말고 0으로 치환하기
        pred_boxes[noobj_mask] = 0
        target_boxes[noobj_mask] = 0

        S = pred_boxes.size(1)
        # 그리드 셀의 idx를 저장용 매쉬 그리드 행렬 생성
        # 이 매쉬 그리드는 글로 설명하긴 어려운데 아무튼 idx을 저장하는 효율적인 행렬이다
        grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
        grid_y = grid_y.to(self.device).float()
        grid_x = grid_x.to(self.device).float()
        # 생성한 매쉬그리드는 CPU에 할당되서 GPU로 이전을 꼭 시켜줘야함

        # 모델의 예측값 tx, ty는 사전에 sigmoid를 적용한다.
        pred_boxes[..., 0] = torch.sigmoid(pred_boxes[..., 0])
        pred_boxes[..., 1] = torch.sigmoid(pred_boxes[..., 1])
        # [tx, ty, tw, th] -> [bx, by, bw, bh]의 실제 변환은 이 함수에서 수행됨
        b_pred_boxes = self.get_pred_bbox(pred_boxes, self.anchor, grid_x, grid_y, S)
        # [t*x, t*y, t*w, t*h] -> [b*x, b*y, b*w, b*h]의 실제 변환은 이 함수에서 수행됨
        b_target_boxes = self.get_pred_bbox(target_boxes, self.anchor, grid_x, grid_y, S)

        return b_pred_boxes, b_target_boxes
    
    # [tx, ty, tw, th] -> [bx, by, bw, bh]변환을 실제로 수행하는 함수
    # [t*x, t*y, t*w, t*h] -> [b*x, b*y, b*w, b*h]변환을 실제로 수행하는 함수
    def get_pred_bbox(self, t_bbox, anchor, grid_x, grid_y, S):
        # 모든 텐서를 동일한 디바이스로 이동
        device = t_bbox.device
        anchor = anchor.to(device)
        grid_x = grid_x.to(device)
        grid_y = grid_y.to(device)

        # 여기에 들어오는 [tx], [ty], [tw], [th]는 모두 [Batch_size, S, S, B] 형태
        tx, ty = t_bbox[..., 0], t_bbox[..., 1]
        tw, th = t_bbox[..., 2], t_bbox[..., 3]

        # S_list를 선언하여 각 S 값에 따른 앵커 박스 인덱스를 설정
        S_list = [52, 26, 13]
        anchor_idx = S_list.index(S)

        # anchor box list에서 cnt 값에 맞춰 객체 크기용 box [3x2]를 가져옴
        pw, ph = anchor[anchor_idx][..., 0], anchor[anchor_idx][..., 1]

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

        return inter_area / union_area
    
def metrics_debug():
    S_list = [52, 26, 13]
    outputs, labels = [], []
    for S in S_list:
        output = torch.rand(1, 255, S, S)
        outputs.append(output)
        label = torch.rand(1, S, S, 255)
        for i in range(3):
            label[..., (i*85)+4:(i*85)+85] = torch.randint(0, 2, (1, S, S, 81), dtype=torch.float32)
        labels.append(label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = YOLOv3Metrics(anchor=anchor_box_list, device=device.type)
    metric_res = metrics.cal_acc_func(outputs, labels)

    iou_score, precision, recall, top1_error = metric_res

    print(f"iou_score: {iou_score}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"top1_error: {top1_error}")



if __name__ == '__main__':
    metrics_debug()