from torch.utils import tensorboard
from data_utils import get_recommendation_metrics, get_recommendation_metrics_multi_label
import torch
from tqdm.auto import tqdm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, 
            model, 
            train_dataloader, 
            test_dataloader, 
            num_epochs=10, 
            alpha=0.5,
            save_dir='models'
        ):
        self.model = model
        self.model.to(device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.writer = tensorboard.SummaryWriter(log_dir='logs')
        self.metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']
        self.save_dir = save_dir
    
    
    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            test_results_ep = {k: 0 for k in self.metrics}
            test_results_sp = {k: 0 for k in self.metrics}
            for batch, entity_labels, stereotype_labels in tqdm(self.test_dataloader, desc="Batches"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                entity_labels = entity_labels.to(device)
                stereotype_labels = stereotype_labels.to(device)

                multi_label = not len(stereotype_labels.shape) == 1

                logits, stp_logits = self.model(input_ids, attention_mask)

                if multi_label:
                    stp_mask = torch.any(stereotype_labels != 0, dim=1)
                    stp_metric_func = get_recommendation_metrics_multi_label
                else:
                    stp_mask = (stereotype_labels != -1)
                    stp_metric_func = get_recommendation_metrics
                    
                
                entity_loss, stp_loss = self.model.get_loss(logits, stp_logits[stp_mask], entity_labels, stereotype_labels[stp_mask])
                
                loss = self.alpha * entity_loss + (1 - self.alpha) * stp_loss

                test_epoch_loss += loss.item()

                test_ep_metrics = get_recommendation_metrics(logits, entity_labels)
                for metric in test_results_ep:
                    test_results_ep[metric] += test_ep_metrics[metric]
                
                test_sp_metrics = stp_metric_func(stp_logits[stp_mask], stereotype_labels[stp_mask])
                for metric in test_results_sp:
                    test_results_sp[metric] += test_sp_metrics[metric]

                # break
            
            avg_test_epoch_loss = test_epoch_loss / len(self.test_dataloader)
            print(f"Test Epoch {epoch} Loss: {avg_test_epoch_loss}")
            self.writer.add_scalar("Test Loss", avg_test_epoch_loss, epoch)


            for metric in test_results_ep:
                test_results_ep[metric] /= len(self.test_dataloader)
                self.writer.add_scalar(f"Test {metric}", test_results_ep[metric], epoch)

            for metric in test_results_sp:
                test_results_sp[metric] /= len(self.test_dataloader)
                self.writer.add_scalar(f"Test {metric}", test_results_sp[metric], epoch)


    def train(self):
        self.model.train()
        min_epoch_loss = float('inf')
        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            epoch_loss = 0
            epoch_results_ep = {k: 0 for k in self.metrics}
            epoch_results_sp = {k: 0 for k in self.metrics}
            
            for batch, entity_labels, stereotype_labels in tqdm(self.train_dataloader, desc="Batches"):
                # print(batch['input_ids'].shape, batch['attention_mask'].shape, entity_labels.shape, stereotype_labels.shape)
                
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                entity_labels = entity_labels.to(device)
                stereotype_labels = stereotype_labels.to(device)

                multi_label = not len(stereotype_labels.shape) == 1

                logits, stp_logits = self.model(input_ids, attention_mask)

                if multi_label:
                    stp_mask = torch.any(stereotype_labels != 0, dim=1)
                    stp_metric_func = get_recommendation_metrics_multi_label
                else:
                    stp_mask = (stereotype_labels != -1)
                    stp_metric_func = get_recommendation_metrics
                    
                
                entity_loss, stp_loss = self.model.get_loss(logits, stp_logits[stp_mask], entity_labels, stereotype_labels[stp_mask])
                
                loss = self.alpha * entity_loss + (1 - self.alpha) * stp_loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

                train_ep_metrics = get_recommendation_metrics(logits, entity_labels)
                for metric in epoch_results_ep:
                    epoch_results_ep[metric] += train_ep_metrics[metric]
                
                train_sp_metrics = stp_metric_func(stp_logits[stp_mask], stereotype_labels[stp_mask])
                for metric in epoch_results_sp:
                    epoch_results_sp[metric] += train_sp_metrics[metric]
                
                # break
                
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch} Loss: {avg_epoch_loss}")
            self.writer.add_scalar("Train Loss", avg_epoch_loss, epoch)


            for metric in epoch_results_ep:
                epoch_results_ep[metric] /= len(self.train_dataloader)
                self.writer.add_scalar(f"Train {metric}", epoch_results_ep[metric], epoch)

            for metric in epoch_results_sp:
                epoch_results_sp[metric] /= len(self.train_dataloader)
                self.writer.add_scalar(f"Train {metric}", epoch_results_sp[metric], epoch)

            if avg_epoch_loss < min_epoch_loss:
                min_epoch_loss = avg_epoch_loss
                torch.save(self.model.state_dict(), self.save_dir + f'/XML4UML_ckpt_{epoch}.pt')

            if epoch % 1 == 0:
                self.test(epoch)
            
            # break
