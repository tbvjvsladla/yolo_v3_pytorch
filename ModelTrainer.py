import torch
from tqdm import tqdm  # 훈련 진행상황 체크
from yolo_v3_metrics import YOLOv3Metrics, anchor_box_list

class ModelTrainer:
    def __init__(self, epoch_step, B=3, C=80, device='cuda'):
        self.epoch_step = epoch_step
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metrics = YOLOv3Metrics(B=B, C=C, anchor=anchor_box_list, device=self.device)

    def model_train(self, model, data_loader, loss_fn, optimizer_fn, scheduler_fn, processing_device, epoch):
        model.train()  # 모델을 훈련 모드로 설정

        # loss와 accuracy를 계산하기 위한 임시 변수를 생성
        run_size, run_loss = 0, 0
        avg_iou, avg_precision, avg_recall, avg_top1_error = 0, 0, 0, 0
        n = 0  # 값의 개수

        # 특정 에폭일 때만 tqdm 진행상황 바 생성
        if (epoch + 1) % self.epoch_step == 0 or epoch == 0:
            progress_bar = tqdm(data_loader)
        else:
            progress_bar = data_loader

        # label은 리스트이니 labels로 이름변경함
        for image, labels in progress_bar:
            # 입력된 데이터를 먼저 GPU로 이전하기
            image = image.to(processing_device)
            # 라벨이 리스트 정보이기에 한개씩 떼어낸다.
            labels = [label.to(processing_device) for label in labels]

            # 전사 과정 수행
            outputs = model(image)
            outputs = [output.permute(0, 2, 3, 1) for output in outputs]

            loss = 0
            for output, label in zip(outputs, labels):
                loss_element = loss_fn(output, label)
                loss += loss_element

            # backward과정 수행
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

            # 스케줄러 업데이트
            scheduler_fn.step()

            # 현재까지 수행한 loss값을 얻어냄
            run_loss += loss.item() * image.size(0)
            run_size += image.size(0)

            # 성능지표 계산하기 -> 리스트이니 outputs, labels임을 유의하자.
            metric_res = self.metrics.cal_acc_func(outputs, labels)
            iou_score, precision, recall, top1_error = metric_res
            n += 1
            avg_iou = avg_iou + (iou_score - avg_iou) / n
            avg_precision = avg_precision + (precision - avg_precision) / n
            avg_recall = avg_recall + (recall - avg_recall) / n
            avg_top1_error = avg_top1_error + (top1_error - avg_top1_error) / n

            # tqdm bar에 추가 정보 기입
            if (epoch + 1) % self.epoch_step == 0 or epoch == 0:
                desc = (f"[훈련중] Loss: {run_loss / run_size:.2f}, "
                        f"KPI: [{avg_iou:.3f}, {avg_precision:.3f}, "
                        f"{avg_recall:.3f}, {avg_top1_error:.3f}]")
                progress_bar.set_description(desc)

        avg_loss = run_loss / len(data_loader.dataset)
        avg_KPI = [avg_iou, avg_precision, avg_recall, avg_top1_error]
        return avg_loss, avg_KPI

    def model_evaluate(self, model, data_loader, loss_fn, processing_device, epoch):
        model.eval()  # 모델을 평가 모드로 전환
        run_loss = 0
        avg_iou, avg_precision, avg_recall, avg_top1_error = 0, 0, 0, 0
        n = 0  # 값의 개수

        # gradient 업데이트를 방지해주자
        with torch.no_grad():
            if (epoch + 1) % self.epoch_step == 0 or epoch == 0:
                progress_bar = tqdm(data_loader)
            else:
                progress_bar = data_loader

            for image, labels in progress_bar:
                image = image.to(processing_device)
                labels = [label.to(processing_device) for label in labels]

                outputs = model(image)
                outputs = [output.permute(0, 2, 3, 1) for output in outputs]

                loss = 0
                for output, label in zip(outputs, labels):
                    loss_element = loss_fn(output, label)
                    loss += loss_element

                run_loss += loss.item() * image.size(0)

                metric_res = self.metrics.cal_acc_func(outputs, labels)
                iou_score, precision, recall, top1_error = metric_res
                n += 1
                avg_iou = avg_iou + (iou_score - avg_iou) / n
                avg_precision = avg_precision + (precision - avg_precision) / n
                avg_recall = avg_recall + (recall - avg_recall) / n
                avg_top1_error = avg_top1_error + (top1_error - avg_top1_error) / n

        avg_loss = run_loss / len(data_loader.dataset)
        avg_KPI = [avg_iou, avg_precision, avg_recall, avg_top1_error]
        return avg_loss, avg_KPI
    

if __name__ == '__main__':
    pass