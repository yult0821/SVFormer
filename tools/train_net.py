#!/usr/bin/env python3
# modified from https://github.com/facebookresearch/SlowFast
"""Train a video classification model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from timm.utils import NativeScaler

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint_amp as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

try:
    from spikingjelly.activation_based import functional  # spikingjelly > 0.0.0.0.12
except:
    from spikingjelly.clock_driven import functional  # spikingjelly == 0.0.0.0.12

logger = logging.get_logger(__name__)

# torch.autograd.set_detect_anomaly(True)

def train_epoch(
    train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_scaler (scaler): scaler for loss.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels, vindex, meta) in enumerate(train_loader):
#         # for debug 20240411
#         print('original input, ', type(inputs), inputs[0].shape)
#         print('whether exists nan in input, ', torch.isnan(inputs[0]).any())
#         print('whether exists inf in input, ', torch.isinf(inputs[0]).any())
#         print('num of 0s in input, ', inputs[0].numel() - torch.count_nonzero(inputs[0]))
#         print('max value in input, ', torch.max(inputs[0]))
#         print('min value in input, ', torch.min(inputs[0]))
#         print('labels, ', type(labels), labels)
#         print('vindex, ', type(vindex), vindex)
#         print('meta, ', type(meta), meta)
#         print('input(B, C, T, H, W) average across TCHW dimension(B): ', torch.mean(inputs[0], dim=[1,2,3,4]))
#         print('input(B, C, T, H, W) std across TCHW dimension(B): ', torch.std(inputs[0], dim=[1,2,3,4]))
        
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast():
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, meta["boxes"])
            else:
                preds = model(inputs)
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        # scaler => backward and step
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRADIENT, parameters=model.parameters(), create_graph=is_second_order)
        
        # reset the model states when using snn (added by yult 0526)
        if cfg.MODEL.ARCH in ['spikingformer', 'spikingformer_cml', 'sew_resnet', 'uniformer_snn', 'uniformer_psnn', 'uniformer2d_snn', 'uniformer2d_psnn', 'uniformer2d_psnn_act', 'uniformer2d_psnn_norm', 'uniformer2d_psnn_bn', 'uniformer2d_psnn_try', 'uniformer2d_psnn_tri', 'uniformer2d_prsnn', 'rsew_resnet', 'uniformer2d_psnn_convBnLif', 'uniformer2d_psnn_head', 'uniformer2d_psnn_lgf', 'uniformer2d_psnn_qkf', 'uniformer2d_psnn_convqkf', 'uniformer2d_if_lgf', 'uniformer2d_psnn_lgfT', 'qkformer', 'uniformer3d_psnn', 'uniformer3d_psnn_lgf', 'sglformer', 'metaspikformer']:
            functional.reset_net(model)
        
        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # # write to tensorboard format if available. (each iteration)
            # if writer is not None:
            #     writer.add_scalars(
            #         {
            #             "Train/loss": loss,
            #             "Train/lr": lr,
            #             "Train/Top1_err": top1_err,
            #             "Train/Top5_err": top5_err,
            #         },
            #         global_step=data_size * cur_epoch + cur_iter,
            #     )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log current epoch stats.
    ce_avg_loss, ce_lr, ce_top1_err, ce_top5_err = train_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available. (each epoch)
    if writer is not None:
        writer.add_scalars(
            {
                "Train/loss": ce_avg_loss,
                "Train/lr": ce_lr,
                "Train/Top1_err": ce_top1_err,
                "Train/Top5_err": ce_top5_err,
            },
            global_step = cur_epoch,
        )
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        loss_scaler (scaler): scaler for loss.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                #print("preds' shape", preds.shape)
                #print("labels' shape", labels.shape)
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                
                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # # write to tensorboard format if available. (each iteration)
                # if writer is not None:
                #     writer.add_scalars(
                #         {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                #         global_step=len(val_loader) * cur_epoch + cur_iter,
                #     )

            val_meter.update_predictions(preds, labels)

        # reset the model states when using snn (added by yult 0526)
        if cfg.MODEL.ARCH in ['spikingformer', 'spikingformer_cml', 'sew_resnet', 'uniformer_snn', 'uniformer_psnn', 'uniformer2d_snn', 'uniformer2d_psnn', 'uniformer2d_psnn_act', 'uniformer2d_psnn_norm', 'uniformer2d_psnn_bn', 'uniformer2d_psnn_try', 'uniformer2d_psnn_tri', 'uniformer2d_prsnn', 'rsew_resnet', 'uniformer2d_psnn_convBnLif', 'uniformer2d_psnn_head', 'uniformer2d_psnn_lgf', 'uniformer2d_psnn_qkf', 'uniformer2d_psnn_convqkf', 'uniformer2d_if_lgf', 'uniformer2d_psnn_lgfT', 'qkformer', 'uniformer3d_psnn', 'uniformer3d_psnn_lgf', 'sglformer', 'metaspikformer']:
            functional.reset_net(model)
        
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    ce_top1_err, ce_top5_err = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available. (each epoch)
    if writer is not None:
        writer.add_scalars(
            {"Val/Top1_err": ce_top1_err, "Val/Top5_err": ce_top5_err},
            global_step = cur_epoch,
        )
    
    # # to save the best val model
    # if val_meter.min_top1_err < gl_MinValErr:
    #     gl_MinValErr = val_meter.min_top1_err
    #     cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
    
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )
            
    # define curEpoch_Valtop1Err to store the best val model
    curEpoch_Valtop1Err = ce_top1_err  # val_meter.min_top1_err
    val_meter.reset()
    return curEpoch_Valtop1Err


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Loss scaler
    loss_scaler = NativeScaler()

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        loss_scaler,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    
    # a flag to save the best val model
    # global gl_MinValtop1Err 
    gl_MinValtop1Err = 100.0

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Loss scaler
    loss_scaler = NativeScaler()

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, loss_scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    loss_scaler,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, loss_scaler, train_meter, cur_epoch, cfg, writer
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            curEpoch_Valtop1Err = eval_epoch(val_loader, model, val_meter, loss_scaler, cur_epoch, cfg, writer)
            # to save the best val model
            if curEpoch_Valtop1Err < gl_MinValtop1Err:
                gl_MinValtop1Err = curEpoch_Valtop1Err
                cu.save_best_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg, gl_MinValtop1Err)
                logger.info(
                    f"Epoch {cur_epoch} reaches new gl_MinValtop1Err {gl_MinValtop1Err}: "
                    f"Save the corresponding ckpt as best_Valtop1Err_checkpoint.pyth in {cu.get_checkpoint_dir(cfg.OUTPUT_DIR)}"
                )

    if writer is not None:
        writer.close()
