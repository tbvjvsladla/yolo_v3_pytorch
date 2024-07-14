import torch
import torch.nn as nn

class Yolov3Loss(nn.Module):
    def __init__(self, B=3, C=80, device='cuda'):
        super(Yolov3Loss, self).__init__()
        # self.S = S # 그리드셀의 단위(52, 26, 13)
        self.B = B # 이 값은 Anchor box의 개수로 인자 변환
        self.C = C # class종류(Coco 데이터 기준 80)
        self.lambda_coord = 5   # 실험적으로 구한 계수값
        self.lambda_noobj = 0.5 # 실험적으로 구한 계수값
        # Objectness loss 및 Classification Loss는 SSE에서 BCE로 변경
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.device = device # 연산에 필요한 변수를 다 GPU로 올려야 함

    def forward(self, outputs, labels):
        # outputs, labels가 리스트 데이터
        tot_loss = 0

        # outputs와 labels를 GPU로 이전 + output data는 차원정렬 수행
        self.outputs = [output.to(self.device).permute(0, 2, 3, 1) for output in outputs]
        self.labels = [label.to(self.device) for label in labels]
        
        for output, label in zip(self.outputs, self.labels):
            element_loss = self.cal_loss(output, label)
            tot_loss += element_loss

        return tot_loss


    def cal_loss(self, predictions, target):
        # Assert 구문을 활용하여 grid cell의 크기가 동일한지 확인
        assert predictions.size(1) == predictions.size(2), "그리드셀 차원이 맞지 않음"

        batch_size = predictions.size(0) # Batch_size의 사이즈 정보를 추출
        S = predictions.size(2) # grid cell의 개수 정보를 추출
        # [Batch_size, S, S, (5 + C) * B] ->  [Batch_size, S, S, B, (5 + C)]로 변환
        predictions = predictions.view(batch_size, S, S, self.B, 5 + self.C)
        target = target.view(batch_size, S, S, self.B, 5 + self.C)

        # 2) BBox좌표, OS, CS 정보 추출하여 재배치
        pred_boxes = predictions[..., :4] # [batch_size, S, S, B, 4]
        # target_boxex의 추출된 정보는 [sigtx, sig, ty, tw, th]이니
        # 이것에 맞게 pred_boxe의 [tx, ty]도 [sigmoid(tx), sigmoid(ty)]처리
        pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2])
        # pred_boxes[..., :2] = torch.clamp(pred_boxes[..., :2], self.e, 1-self.e)
        target_boxes = target[..., :4] # [batch_size, S, S, B, 4]

        # Objectness Score에 sigmoid 적용 -> 이게 Yolo v3의 핵심임
        pred_obj = torch.sigmoid(predictions[..., 4])
        target_obj = target[..., 4]    # [batch_size, S, S, B]

        # Class scores에 sigmoid 적용 -> 이게 Yolo v3의 핵심임
        pred_class = torch.sigmoid(predictions[..., 5:])
        target_class = target[..., 5:] # [batch_size, S, S, B, C]

        # 3) 마스크 필터 생성
        coord_mask = target_obj > 0 # [batch_size, S, S, B]
        noobj_mask = target_obj <= 0 # [batch_size, S, S, B]
        # 예외처리 : 객체가 아에 없는 이미지에 대한
        if coord_mask.sum() == 0 and noobj_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        # 4) 마스크 필터를 적용해 Bbox의 데이터 필터링
        # unsqueeze를 이용해 [batch_size, S, S, B] -> [batch_size, S, S, B, 1]
        # 이후 expand_as를 이용해 [batch_size, S, S, B, 4]
        # expand_as는 expand랑 거의 기능이 같다
        coord_mask_box = coord_mask.unsqueeze(-1).expand_as(target_boxes)

        # 5) Localization Loss 계산하기
        # bbox의 값이 [tx, ty, tw, th]이니 한꺼번에 loss 계산
        coord_loss = self.lambda_coord * (
            (pred_boxes[coord_mask_box] - target_boxes[coord_mask_box]) ** 2
        ).sum()

        # 6) OS(objectness Score)의 마스크 필터를 활용한 필터링
        # 7) Objectness Loss 계산하기
        obj_loss = self.bce_loss(pred_obj[coord_mask], target_obj[coord_mask])
        
        cal = self.bce_loss(pred_obj[noobj_mask], target_obj[noobj_mask])
        noobj_loss = self.lambda_noobj * cal

        # 8) CP(Class Probabilities)에 대한 필터링 작업
        # 9) Classification Loss 계산하기
        # coord_mask는 자동으로 broadcasting이 되어서 expand_as를 안써도 된다.
        class_loss = self.bce_loss(pred_class[coord_mask], target_class[coord_mask])

        element_loss = coord_loss + obj_loss + noobj_loss + class_loss

        return element_loss
    

def loss_debug():
    # 디버그용 데이터 생성하기
    S_list = [52, 26, 13]
    outputs, targets = [], []
    for S in S_list:
        output = torch.rand(1, 255, S, S)
        outputs.append(output)
        target = torch.rand(1, S, S, 255)
        for i in range(3):
            target[..., (i*85)+4:(i*85)+85] = torch.randint(0, 2, (1, S, S, 81), dtype=torch.float32)
        targets.append(target)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yolov3_loss = Yolov3Loss(B=3, C=80, device=device.type)

    loss = yolov3_loss(outputs, targets)
    print(f"Loss: {loss.item()}")


if __name__ == '__main__':
    loss_debug()
