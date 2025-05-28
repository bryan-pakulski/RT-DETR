"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils
from ..misc.dist_utils import gprint

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

def prepare_eval_metric(metric, catId, type):
    # type precision: (iou, recall, cls, area range, max dets)
    # type recall: (iou, cls, area range, max dets)
    if type == "precision":
        metric = [metric[i][j][catId][0][-1] for i in range(len(metric)) for j in range(len(metric[i]))]
    elif type == "recall":
        metric = [metric[i][catId][0][-1] for i in range(len(metric))]

    # Filter out values <= -1
    filtered_metric = [value for value in metric if value > -1]

    # Calculate mean or return NaN if empty
    if filtered_metric:
        return sum(filtered_metric) / len(filtered_metric)
    else:
        return float("nan")

class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")

        # This sets up config, data loaders, optimizers, ema etc...
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # NOTE: when using dynamic batch sizing for mixed gpu training our datasets can have different
            # lengths due to batch size where number_of_train_iterations = dataset_size / batch_size which
            # will cause hanging while we wait for each GPU to finish training and the first gpu to finish is calling for a sync
            # We subset the dataset into mini batches if required so each GPU has the same number of samples during training
            # NOTE: for validation we don't want to split into minibatches as we want to make sure that we are fully evaluating the model
            if (dist_utils.is_parallel(self.model) and len(args.device_batch_split) > 0):
                dist_utils.gprint("Building minibatch subset for training...")
                train_subset = dist_utils.subset_dataset_by_rank(args, self.train_dataloader, args._train_dataloader, args._train_shuffle)
            else:
                train_subset = self.train_dataloader

            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                train_subset, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            coco_eval = coco_evaluator.coco_eval["bbox"]
            precisions = coco_evaluator.coco_eval["bbox"].eval['precision']
            recalls = coco_evaluator.coco_eval["bbox"].eval['recall']

            class_results = {}
            for category_id in coco_eval.cocoGt.getCatIds():
                category_info = coco_eval.cocoGt.loadCats([category_id])[0]
                category_name = category_info['name']
                ap = prepare_eval_metric(precisions, category_id - 1, "precision")
                ar = prepare_eval_metric(recalls, category_id - 1, "recall")
                class_results[category_id] = {'name': category_name, 'mAP': ap, 'mAR': ar}

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                    for class_id in class_results.keys():
                        self.writer.add_scalar(f'Test/class_{class_results[class_id]["name"]}_mAP', class_results[class_id]["mAP"], epoch)
                        self.writer.add_scalar(f'Test/class_{class_results[class_id]["name"]}_mAR', class_results[class_id]["mAR"], epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
