import torch
from torch.utils.data import DataLoader
import M2TR.models 
import M2TR.datasets
import M2TR.utils.distributed as du
import M2TR.utils.logging as logging
from M2TR.utils.registries import (
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
)

logger = logging.get_logger(__name__)


def build_model(cfg, gpu_id=None):
    # Construct the model
    model_cfg = cfg['MODEL']
    name = model_cfg['MODEL_NAME']
    logger.info('MODEL_NAME: ' + name)
    model = MODEL_REGISTRY.get(name)(model_cfg)

    # Determine the device to use (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    elif torch.backends.mps.is_available():
        print("MPS backend is available. Using MPS.")
        device = torch.device("mps")
    else:
        print("CUDA and MPS are not available. Using CPU.")
        device = torch.device("cpu")

    # Log the chosen device
    logger.info(f"Using device: {device}")

    # Check GPU availability and adjust accordingly
    assert (
        cfg['NUM_GPUS'] <= torch.cuda.device_count() if torch.cuda.is_available() else 1
    ), "Cannot use more GPU devices than available"

    # Move the model to the appropriate device
    model = model.to(device)

    # Use multi-process data parallel model in the multi-GPU setting
    if cfg['NUM_GPUS'] > 1 and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[gpu_id] if gpu_id is not None else [torch.cuda.current_device()], 
            output_device=gpu_id if gpu_id is not None else torch.cuda.current_device(),
            find_unused_parameters=True
        )

    return model


def build_loss_fun(cfg):
    loss_cfg = cfg['LOSS']
    name = loss_cfg['LOSS_FUN']
    logger.info('LOSS_FUN: ' + name)
    loss_fun = LOSS_REGISTRY.get(name)(loss_cfg)
    return loss_fun


def build_dataset(mode, cfg):
    dataset_cfg = cfg['DATASET']
    name = dataset_cfg['DATASET_NAME']
    logger.info('DATASET_NAME: ' + name + '  ' + mode)
    return DATASET_REGISTRY.get(name)(dataset_cfg, mode)


def build_dataloader(dataset, mode, cfg):
    dataloader_cfg = cfg['DATALOADER']
    num_tasks = du.get_world_size()
    global_rank = du.get_rank()

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True if mode == 'train' else False,
    )

    return DataLoader(
        dataset,
        batch_size=dataloader_cfg['BATCH_SIZE'],
        sampler=sampler,
        num_workers=dataloader_cfg['NUM_WORKERS'],
        pin_memory=dataloader_cfg['PIN_MEM'],
        drop_last=True if mode == 'train' else False,
    )
