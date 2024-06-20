import datetime
import os
import sys
import time
from loguru import logger
import wandb

import torch
import torch.nn as nn

from finetuning.experiments import SAMExp
import torch.nn.functional as F
from tqdm import tqdm
from statistics import mean
import monai


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer:
    def __init__(self, param_cls: SAMExp, args):
        self.sam_exp = param_cls
        self.args = args

        self.max_epoch = self.sam_exp.max_epoch
        self.max_lr = self.sam_exp.max_lr
        self.learning_rate = self.sam_exp.lr
        self.grad_clip = self.sam_exp.grad_clip
        self.weight_decay = self.sam_exp.weight_decay

        self.save_history_ckpt = self.sam_exp.save_history_ckpt

        # self.save_name = os.path.join(self.sam_exp.output_dir, args.experiment_name)

        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.device = self.sam_exp.device
        self.loss = self.sam_exp.get_loss_function(self.sam_exp.loss)
        # self.loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        
        # Initialization of accuracies and losses
        self.best_acc = 0
        self.epoch_acc = 0
        self.val_loss = 0
        self.return_predictions = self.sam_exp.return_predictions

        # setup_logger(
        #     self.file_name,
        #     filename="train_log.txt",
        #     mode="a",
        # )

    def train(self):
        self.before_train()
        try:
            self.train_epochs()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        logger.info("args: {}".format(self.args)) 
        logger.info("Basic class value:\n{}".format(self.sam_exp))

        # model related init
        self.model = self.sam_exp.get_model()
        self.model.to(self.device)

        from torchsummary import summary
        logger.info(
            "Model Summary: {}".format(self.model)
        )
        
        # data related init
        self.train_loader = self.sam_exp.get_train_loader(
            batch_size=self.args.batch_size
        )

        # solver related init
        self.optimizer = self.sam_exp.optimizer(self.model.parameters(), lr=self.max_lr, weight_decay=self.weight_decay)
        # self.lr_scheduler = self.sam_exp.scheduler(self.optimizer, max_lr=self.max_lr, epochs=self.max_epoch, steps_per_epoch=len(self.train_loader))
        # self.lr_scheduler = self.sam_exp.scheduler(self.optimizer, base_lr=0.001, max_lr=0.02,step_size_up=1000,mode="exp_range",gamma=0.9995, cycle_momentum=False)

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        if self.args.logger == "wandb":
            # Wandb logger
            config = {
                "batch_size": self.args.batch_size,
                "max_epoch": self.sam_exp.max_epoch,
                "weight_decay": self.sam_exp.weight_decay,
                "sam_type": self.sam_exp.sam_type,
                "optimizer": self.optimizer
            }

            self.wandb_logger = wandb.init(project=self.sam_exp.project_name,
                                       config=config)
            
            # Metric record
            # self.wandb_logger.define_metric("train_step")
            # self.wandb_logger.define_metric("train/*", step_metric="train_step")

            # self.wandb_logger.define_metric("val_step")
            # self.wandb_logger.define_metric("val/val_loss", step_metric="val_step")

            self.file_name = os.path.join(self.sam_exp.output_dir, self.sam_exp.project_name, self.args.experiment_name, self.wandb_logger.name)
            os.makedirs(self.file_name, exist_ok=True)

        else:
            raise ValueError("logger must be 'wandb'")
                
        logger.info("Starting the training process ...")
        logger.info("\n{}".format(self.model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the highest precision is {:.2f}".format(self.best_acc * 100)
        )

        if self.args.logger == "wandb":
            self.wandb_logger.finish()

    def train_epochs(self):
        self.model.train()
        self.wandb_step = 0
        loss_epoch = []
        for self.epoch in range(self.max_epoch):
            # Before epoch
            logger.info(" *** Start training epoch {} ***".format(self.epoch + 1))

            # self.model.train()
            train_losses = []
            lrs = []

            self.iter_step = 0
            loss_iter = []
            for batch in self.train_loader:
                self.wandb_step += 1
                self.iter_step += 1
                iter_start_time = time.time()

                input_image = batch["image"].to(self.device)
                ground_truth_masks = batch["ground_truth_masks"]
                bboxes = batch["bboxes"]


                input_size = tuple(batch["input_size"].numpy()[0])
                original_image_size = tuple(batch["original_image_size"].numpy()[0])

                # We do not finetune the encoder
                with torch.no_grad():
                    image_embedding = self.model.image_encoder(input_image)

                loss_masks = []
                for i, gt_mask in enumerate(ground_truth_masks):
                    gt_mask = gt_mask.unsqueeze(0).to(self.device)
                    bbox = bboxes[i].to(self.device)

                    with torch.no_grad():
                        sparse_bbox_embeddings, dense_bbox_embeddings = self.model.prompt_encoder(
                            points=None,
                            boxes=bbox,
                            masks=None
                        )
                    
                    predicted_mask, iou_predictions = self.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_bbox_embeddings,
                        dense_prompt_embeddings=dense_bbox_embeddings,
                        multimask_output=False
                    )

                    predicted_mask = torch.sigmoid(predicted_mask)

                    # Calculate loss
                    loss_mask = self.loss(predicted_mask, gt_mask.float())

                    loss_mask.backward()

                    # Losses of all masks of one image
                    loss_masks.append(loss_mask.item())

                self.optimizer.zero_grad()
                self.optimizer.step()
                # self.lr_scheduler.step()

                # Batch loss is the mean of image-masks losses
                loss_iter.append(mean(loss_masks))
            
                iter_end_time = time.time()
                iter_time = iter_end_time - iter_start_time

                if self.args.logger == "wandb":
                    self.wandb_logger.log({"train/loss": mean(loss_iter),
                                           "train/learning_rate": self.learning_rate}, # self.lr_scheduler.get_last_lr()[0]
                                           step=self.wandb_step)

                logger.info("Epoch: {}/{}, iter: {}/{}, iter_loss: {}, training time: {} s".format(self.epoch+1, self.max_epoch, self.iter_step, self.max_iter, mean(loss_iter), iter_time))

            # Epoch loss is average of all batches losses
            loss_epoch.append(mean(loss_iter))
                    
            # # After epoch
            # self.save_ckpt(ckpt_name="latest")
            # if (self.epoch + 1) % self.sam_exp.eval_interval == 0:
            #     self.evaluate_and_save_model()
            #     if self.args.logger == "wandb":
            #         self.wandb_logger.log_metrics({"val/val_accuracy": self.epoch_acc.item(), 
            #                                         "val/val_loss": self.val_loss.item()}, step=self.wandb_step)

            # After epoch
            if self.save_history_ckpt:
                self.save_ckpt(f"epoch_{self.epoch + 1}", acc=self.epoch_acc)

    def evaluate_and_save_model(self):
        evalmodel = self.model
        self.val_loader = self.sam_exp.get_eval_data_loader(
            batch_size=self.args.batch_size,
            device=self.device
        )
        
        output_dict = self.sam_exp.eval(evalmodel, self.val_loader, self.device)
        
        self.epoch_acc = output_dict["val_accuracy"]
        self.val_loss = output_dict["val_loss"]

        update_best_ckpt = self.epoch_acc > self.best_acc
        self.best_acc = max(self.best_acc, self.epoch_acc)

        logger.info("Epoch accuracy: {}".format(self.epoch_acc))
        logger.info("Best accuracy: {}".format(self.best_acc))
        self.save_ckpt("last_epoch", update_best_ckpt, acc=self.epoch_acc)

        logger.info(" *** Training of epoch {} ended! Epoch accuracy: {}, best training accuracy achieved: {} ***".format(self.epoch + 1, self.epoch_acc, self.best_acc))

        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", acc=self.epoch_acc)

        
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, acc=None):

        save_model = self.model
        logger.info("Saving weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_ap": self.best_acc,
            "curr_ap": acc,
        }
        filename = os.path.join(self.file_name, ckpt_name + "_ckpt.pth")
        torch.save(ckpt_state, filename)

        if update_best_ckpt:
            filename = os.path.join(self.file_name, "best_ckpt.pth")
            torch.save(ckpt_state, filename)