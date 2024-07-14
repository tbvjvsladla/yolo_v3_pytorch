import torch
from tqdm import tqdm  # 훈련 진행상황 체크

class ModelTrainer:
    def __init__(self, epoch_step, device='cuda'):
        self.epoch_step = epoch_step
        self.device = device # 연산에 필요한 변수를 다 GPU로 올려야 함

    def update_metrics(self, outputs, labels, metric_fn, metrics, n):
        metric_res = metric_fn.cal_acc_func(outputs, labels)
        # iou_score, precision, recall, top1_error로 데이터가 출력됨
        i, p, r, t = metric_res
        n += 1
        # 이동평균 필터로 현재 평가지표의 평균값을 업데이트
        metrics['I'] = metrics['I'] + (i - metrics['I']) / n
        metrics['P'] = metrics['P'] + (p - metrics['P']) / n
        metrics['R'] = metrics['R'] + (r - metrics['R']) / n
        metrics['T'] = metrics['T'] + (t - metrics['T']) / n
        return metrics, n

    def model_train(self, model, data_loader, 
                    loss_fn, optimizer_fn, scheduler_fn, metric_fn,
                    epoch):
        model.train()  # 모델을 훈련 모드로 설정

        # loss와 accuracy를 계산하기 위한 임시 변수를 생성
        run_size, run_loss = 0, 0
        # I = IOU, P = Precision, R = Recall, T = Top-1_error
        run_metrics = {'I': 0, 'P': 0, 'R': 0, 'T': 0}
        n = 0  # 값의 개수

        # 특정 에폭일 때만 tqdm 진행상황 바 생성
        if (epoch + 1) % self.epoch_step == 0 or epoch == 0:
            progress_bar = tqdm(data_loader)
        else:
            progress_bar = data_loader

        # label은 리스트이니 labels로 이름변경함
        for image, labels in progress_bar:
            # 입력된 데이터를 먼저 GPU로 이전하기
            image = image.to(self.device)
            # 라벨이 리스트 정보이기에 한개씩 떼어낸다.
            labels = [label.to(self.device) for label in labels]

            # 전사 과정 수행
            outputs = model(image)
            loss = loss_fn(outputs, labels)

            # backward과정 수행
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()

            # 스케줄러 업데이트
            scheduler_fn.step()

            # 현재까지 수행한 loss값을 얻어냄
            run_loss += loss.item() * image.size(0)
            # desc에 진행상황 체크용 분모값
            run_size += image.size(0)

            # 성능지표 계산하기 -> 리스트이니 outputs, labels임을 유의하자.
            run_metrics, n = self.update_metrics(outputs, labels, 
                                                 metric_fn, 
                                                 run_metrics, n)

            # tqdm bar에 추가 정보 기입
            if (epoch + 1) % self.epoch_step == 0 or epoch == 0:
                desc = (f"[훈련중] Loss: {run_loss / run_size:.2f}, "
                        f"KPI: [{run_metrics['I']:.3f}, {run_metrics['P']:.3f}, "
                        f"{run_metrics['R']:.3f}, {run_metrics['T']:.3f}]")
                progress_bar.set_description(desc)

        avg_loss = run_loss / len(data_loader.dataset)
        avg_KPI = [val for key, val in run_metrics.items()]
        return avg_loss, avg_KPI

    def model_evaluate(self, model, data_loader, 
                       loss_fn, metric_fn,
                       epoch):
        model.eval()  # 모델을 평가 모드로 전환
        run_loss = 0
        # I = IOU, P = Precision, R = Recall, T = Top-1_error
        run_metrics = {'I': 0, 'P': 0, 'R': 0, 'T': 0}
        n = 0  # 값의 개수

        # gradient 업데이트를 방지해주자
        with torch.no_grad():
            if (epoch + 1) % self.epoch_step == 0 or epoch == 0:
                progress_bar = tqdm(data_loader)
            else:
                progress_bar = data_loader

            for image, labels in progress_bar:
                image = image.to(self.device)
                labels = [label.to(self.device) for label in labels]

                # 전사과정 수행
                outputs = model(image)
                loss = loss_fn(outputs, labels)
                # 현재까지 수행한 loss값을 얻어냄
                run_loss += loss.item() * image.size(0)

                # 성능지표 계산하기 -> 리스트이니 outputs, labels임을 유의하자.
                run_metrics, n = self.update_metrics(outputs, labels, 
                                                     metric_fn, 
                                                     run_metrics, n)

        avg_loss = run_loss / len(data_loader.dataset)
        avg_KPI = [val for key, val in run_metrics.items()]
        return avg_loss, avg_KPI
    

if __name__ == '__main__':
    pass