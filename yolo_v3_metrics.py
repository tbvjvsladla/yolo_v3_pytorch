import torch
import torch.nn as nn
import torch.nn.functional as F
import coco_data #데이터 관리용 py파일
from yolo_util import YoloComFunc #좌표변환, IOU연산 클래스

class YOLOv3Metrics(YoloComFunc):
    def __init__(self, B=3, C=80, device='cuda',
                 anchor=coco_data.anchor_box_list):
        super().__init__(anchor=anchor) # 이 anchor 값을 부모 클래스에 전달
        self.B = B
        self.C = C
        self.device = device # 연산에 필요한 변수를 다 GPU로 올려야 함

    # 2, 3번 평가지표 계산(mAP, Recall에 mean처리한 값)
    def cal_P_R_curve(self, ious, pred_os, target_os, bs):
        # pred_os의 shape =  #[batch_size, S, S, B]
        ious = ious.reshape(bs, -1) #[batch_size, S*S*B]
        pred_os = pred_os.reshape(bs, -1) #[batch_size, S*S*B]
        target_os = target_os.reshape(bs, -1) #[batch_size, S*S*B]

        iou_ths = torch.linspace(0.5, 0.95, 10).to(self.device)
        # (10) -> (batch_size, S*S*B, 10)으로 iou확장
        ious = ious.unsqueeze(-1).expand(-1, -1, 10)
        #iou의 차원을 ious와 같게 만들기
        iou_ths = iou_ths.view(1, 1, -1).expand_as(ious)
        # true_positives 계산 (batch_size, S*S*B, 10)
        TP = (ious >= iou_ths).float()

        # 각 IoU 임계값에 대해 정렬 수행
        s_indices = torch.argsort(-pred_os, dim=-1)  #(batch_size, S*S*B)
        s_indices = s_indices.unsqueeze(-1).expand_as(TP)  #(batch_size, S*S*B, 10)
        
        # 정렬된 신뢰도값으로 True Positive를 정렬 후 차원 축소
        s_TP = torch.gather(TP, 1, s_indices)

        # 누적합 계산 및 Precision / Recall 계산
        tp_cumsum = torch.cumsum(s_TP, dim=1) # S*S*B 차원에서 누적합 계산 -> (batch_size, S*S*B, 10)
        precision = tp_cumsum / (torch.arange(tp_cumsum.size(1)).to(self.device).float() + 1).view(1, -1, 1)
        recall = tp_cumsum / (target_os.view(bs, -1).sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-16)

        # mAP 계산
        mAP = torch.trapz(precision, recall, dim=1).mean()
        m_recall = recall.mean()
        # recall은 그냥 평균값을 내서 보냄
        return mAP, m_recall

    # 1번 평가지표 : True Positive에 대한 IOU
    def cal_tp_iou(self, pred_b_bbox, target_b_bbox, coord_mask):
        ious = self.compute_iou(pred_b_bbox, target_b_bbox)
        # 전체 IOU 항목 중 True Positive 항목만 필터링
        TP_ious = ious[coord_mask]
        return ious, TP_ious.mean()

    # 4번 평가지표 : top-1 error
    def cal_top_1_error(self, pred_cp, target_cp, coord_mask, S):
        # 객체 존재지역을 [batch_size, S*S*B] 차원으로 재정렬
        coord_mask = coord_mask.reshape(-1, S*S*self.B)
        #[batch_size, S, S, B, C] -> [batch_size, S*S*B, C]
        pred_cp = pred_cp.reshape(-1, S*S*self.B, self.C)
        target_cp = target_cp.reshape(-1, S*S*self.B, self.C)
        # Softmax 적용 -> class 차원인 C항목에 적용함
        pred_cp = F.softmax(pred_cp, dim=-1)

        # Class 확률 정보에서 가장 높은 값을 갖는 idx 정보 추출
        pred_cls = torch.argmax(pred_cp, dim=-1)
        target_cls = torch.argmax(target_cp, dim=-1)

        # 객체가 존재하는 지역으로 필터링 수행
        f_pred_cls = pred_cls[coord_mask]
        f_target_cls = target_cls[coord_mask]
        # top-1_error 연산
        top1_error = (~f_pred_cls.eq(f_target_cls))
        return top1_error.float().mean()

    # 원소값 기준으로 평가지표를 연산하는 함수
    def cal_element_metrics(self, pred, target):
        # Assert 구문을 활용하여 grid cell의 크기가 동일한지 확인
        assert pred.size(1) == target.size(1), "데이터 정렬 오류"

        # batch_size, S의 크기 확인
        bs, S = pred.size(0), pred.size(1)

        # [batch_size, S, S, B*(C+5)]-> [batch_size, S, S, B, (C+5)]
        pred = pred.view(bs, S, S, self.B, 5 + self.C)
        target = target.view(bs, S, S, self.B, 5 + self.C)

        # tx, ty, OS(objectness score), 항목에 sigmoid 적용
        pred[..., :2] = torch.sigmoid(pred[..., :2])  # t_x, t_y
        pred[..., 4] = torch.sigmoid(pred[..., 4])  # OS

        # bbox, os, cp로 데이터 분해
        pred_t_bbox = pred[..., :4] #[batch_size, S, S, B, 4]
        target_t_bbox = target[..., :4] #[batch_size, S, S, B, 4]
        pred_os = pred[..., 4] #[batch_size, S, S, B]
        target_os = target[..., 4] #[batch_size, S, S, B]
        pred_cp = pred[..., 5:] #[batch_size, S, S, B, C]
        target_cp = target[..., 5:] #[batch_size, S, S, B, C]

        # obj 정보로 마스크 필터 생성하기
        coord_mask = target_os > 0 #객체가 있는 지역 -> coord_mask
        if coord_mask.sum() == 0: #객체가 아에 없는 예외처리
            return torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        noobj_mask = target_os <= 0

        # [tx, ty, tw, th] -> [bx, by, bw, bh] 변환 실행
        pred_b_bbox = self.get_pred_bbox(pred_t_bbox)
        target_b_bbox = self.get_pred_bbox(target_t_bbox)
        # 마스크 필터로 target의 bbox 데이터 필터링
        # os가 0이면 t_series_bbox가 0인데, 해당 함수를 통과하면
        # b_series_bbox가 [grid/s, gird/s, pw, ph]값이 차버리게 됨
        target_b_bbox[noobj_mask] = 0

        # 1번 평가지표 Ture Positive만의 IOU값 계산
        ious, TP_iou = self.cal_tp_iou(pred_b_bbox, target_b_bbox, coord_mask)
        # 4번 평가지표 원소 Top-1 Error 계산
        top1_err = self.cal_top_1_error(pred_cp, target_cp, coord_mask, S)
        # 2, 3번 평가지표 mAP, Recall의 평균값 계산
        mAP, m_recall = self.cal_P_R_curve(ious, pred_os, target_os, bs)

        #원소 평가지표 4가지를 리턴
        element = torch.stack([TP_iou, mAP, m_recall, top1_err], dim= -1)
        return element

    # mini_batch에 대한 평가지표 산출함수
    def cal_acc_func(self, outputs, labels):
        # outputs와 labels를 GPU로 이전 + output의 데이터 정렬 수행
        self.outputs = [output.to(self.device).permute(0, 2, 3, 1) for output in outputs]
        self.labels = [label.to(self.device) for label in labels]

        res_metrics = []

        # 리스트 내 원소들을 추출한 후 연산 수행
        for output, label in zip(self.outputs, self.labels):
            # 실제 원소별 평가지표 연산은 `compute_metrics`함수에서 수행
            element_metrics = self.cal_element_metrics(output, label)
            res_metrics.append(element_metrics)

        # 리스트를 텐서로 변환하여 평균 계산
        res_metrics = torch.stack(res_metrics)
        mean_metrics = res_metrics.mean(dim=0)

        cal_res = [mean_metrics[i].item() for i in range(mean_metrics.size(0))]
        return cal_res


def metrics_debug():
    S_list = coco_data.S_list
    outputs, labels = [], []
    for S in S_list:
        output = torch.rand(1, 255, S, S)
        outputs.append(output)
        label = torch.rand(1, S, S, 255)
        for i in range(3):
            label[..., (i*85)+4:(i*85)+85] = torch.randint(0, 2, (1, S, S, 81), dtype=torch.float32)
        labels.append(label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = YOLOv3Metrics(anchor=coco_data.anchor_box_list, device=device.type)
    metric_res = metrics.cal_acc_func(outputs, labels)

    iou_score, precision, recall, top1_error = metric_res

    print(f"iou_score: {iou_score}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"top1_error: {top1_error}")



if __name__ == '__main__':
    metrics_debug()