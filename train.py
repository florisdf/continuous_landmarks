import argparse
from itertools import chain
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import wandb

from continuous_landmarks.dataset import face300w, facescape, fitymi, concat
from continuous_landmarks.model import FeatureExtractor, LandmarkPredictor,\
    PositionEncoder
from continuous_landmarks.utils.kfold import kfold_split
from continuous_landmarks.dataset.transforms import (
    Compose, Align, CenterCrop, Resize, RandomResizedCrop,
    RandomRotation, ColorJitter, ToTensor, Normalize,
    AbsToRelLdmks,
)
from continuous_landmarks.training import TrainingLoop, TrainingSteps


def run_training(
    # Model
    feat_model,
    lm_model,

    # Dataset
    data_path_300w,
    data_path_fitymi,
    data_path_facescape,

    # Data augmentations
    input_size,
    rrc_scale,
    rrc_ratio,
    random_angle,
    random_brightness,
    random_contrast,
    random_saturation,
    norm_mean,
    norm_std,

    # Ckpt
    load_ckpt,
    no_save_ckpts,
    best_metric,
    higher_is_better,
    ckpts_path,

    # K-Fold
    k_fold_seed,
    k_fold_num_folds,
    k_fold_val_fold,

    # Dataloader
    batch_size,
    val_batch_size,
    num_workers,

    # Optimizer
    lr,
    beta1,
    beta2,
    weight_decay,
    lr_warmup_steps,

    # Train
    num_epochs,
    val_every,

    # Device
    device,
):
    dl_train, dl_val_300w, dl_val_fitymi, dl_val_facescape = \
        get_data_loaders(
            data_path_300w, data_path_fitymi, data_path_facescape,
            k_fold_num_folds, k_fold_val_fold, k_fold_seed,
            batch_size, val_batch_size, num_workers,
            input_size, rrc_scale, rrc_ratio,
            random_angle, random_brightness, random_contrast,
            random_saturation, norm_mean, norm_std,
        )

    device = torch.device(device)

    pos_encoder = PositionEncoder()
    feat_extractor = FeatureExtractor(feat_model)
    lm_predictor = LandmarkPredictor(
        query_size=pos_encoder.encoding_size,
        feature_size=feat_extractor.feature_size,
        model_name=lm_model,
    )

    if load_ckpt is not None:
        state_dicts = torch.load(load_ckpt)
        pos_encoder.load_state_dict(state_dicts['PositionEncoder'])
        feat_extractor.load_state_dict(state_dicts['FeatureExtractor'])
        lm_predictor.load_state_dict(state_dicts['LandmarkPredictor'])

    training_steps = TrainingSteps(
        pos_encoder=pos_encoder,
        feat_extractor=feat_extractor,
        lm_predictor=lm_predictor,
    )

    optimizer = AdamW(
        chain(pos_encoder.parameters(),
              feat_extractor.parameters(),
              lm_predictor.parameters()),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1/lr_warmup_steps,
        end_factor=1.0,
        total_iters=lr_warmup_steps
    )

    training_loop = TrainingLoop(
        training_steps=training_steps,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=num_epochs,
        dl_train=dl_train,
        dl_val_list=[dl_val_300w, dl_val_fitymi, dl_val_facescape],
        val_every=val_every,
        save_ckpts=not no_save_ckpts,
        best_metric=best_metric,
        higher_is_better=higher_is_better,
        ckpts_path=ckpts_path,
    )
    training_loop.run()


def get_data_loaders(
    data_path_300w, data_path_fitymi, data_path_facescape,
    k_fold_num_folds, k_fold_val_fold, k_fold_seed,
    batch_size, val_batch_size, num_workers,
    input_size, rrc_scale, rrc_ratio,
    random_angle, random_brightness, random_contrast,
    random_saturation, norm_mean, norm_std,
):
    common_train_tfms = [
        RandomResizedCrop(input_size, scale=rrc_scale, ratio=rrc_ratio),
        ColorJitter(random_brightness, random_contrast, random_saturation),
        AbsToRelLdmks(),
        ToTensor(),
        Normalize(norm_mean, norm_std),
    ]
    if random_angle != 0:
        common_train_tfms = [
            RandomRotation(degrees=random_angle),
            *common_train_tfms
        ]

    common_val_tfms = [
        Resize(input_size),
        CenterCrop(input_size),
        AbsToRelLdmks(),
        ToTensor(),
        Normalize(norm_mean, norm_std),
    ]

    # Set up 300W
    data_path_300w = Path(data_path_300w)
    ds_train_300w = face300w.Face300WDataset(
        data_path=data_path_300w,
        transform=Compose([
            Align(face300w.get_eyes_mouth),
            *common_train_tfms
        ]),
    )
    ds_train_300w, ds_val_300w = kfold_split(
        ds_train_300w,
        k=k_fold_num_folds,
        val_fold=k_fold_val_fold,
        seed=k_fold_seed,
    )
    ds_val_300w.transform = Compose([
        Align(face300w.get_eyes_mouth),
        *common_val_tfms
    ])

    # Set up FITYMI
    data_path_fitymi = Path(data_path_fitymi)
    ds_train_fitymi = fitymi.FITYMIDataset(
        data_path=data_path_fitymi,
        transform=Compose([
            Align(fitymi.get_eyes_mouth),
            *common_train_tfms
        ]),
    )
    ds_train_fitymi, ds_val_fitymi = kfold_split(
        ds_train_fitymi,
        k=k_fold_num_folds,
        val_fold=k_fold_val_fold,
        seed=k_fold_seed,
    )
    ds_val_fitymi.transform = Compose([
        Align(fitymi.get_eyes_mouth),
        *common_val_tfms
    ])

    # Set up FaceScape
    data_path_facescape = Path(data_path_facescape)
    ds_train_facescape = facescape.FaceScapeLandmarkDataset(
        data_path=data_path_facescape,
        transform=Compose([
            Align(facescape.get_eyes_mouth),
            *common_train_tfms
        ]),
    )
    ds_train_facescape, ds_val_facescape = kfold_split(
        ds_train_facescape,
        k=k_fold_num_folds,
        val_fold=k_fold_val_fold,
        seed=k_fold_seed,
    )
    ds_val_facescape.transform = Compose([
        Align(facescape.get_eyes_mouth),
        *common_val_tfms
    ])
    ds_train_facescape.df = ds_train_facescape.df.sample(
        min(len(ds_train_fitymi), len(ds_train_facescape))
    ).reset_index(drop=True)
    ds_val_facescape.df = ds_val_facescape.df.sample(
        min(len(ds_val_fitymi), len(ds_val_facescape)),
        random_state=42,
    ).reset_index(drop=True)

    # Create training set by concatenating the different training sets
    ds_train = concat.ConcatDataset([ds_train_300w, ds_train_fitymi,
                                     ds_train_facescape])
    dl_train = DataLoader(
        ds_train,
        num_workers=num_workers,
        batch_sampler=concat.ConcatBatchSampler(
            concat_dataset=ds_train,
            batch_size=batch_size,
            shuffle=True,
        )
    )

    # Validation data loaders
    dl_val_300w = DataLoader(
        ds_val_300w,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )
    dl_val_fitymi = DataLoader(
        ds_val_fitymi,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )
    dl_val_facescape = DataLoader(
        ds_val_facescape,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )

    return dl_train, dl_val_300w, dl_val_fitymi, dl_val_facescape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument(
        '--feat_model', default='ConvNeXt',
        help="The feature extractor to use.",
    )
    parser.add_argument(
        '--lm_model', default='Transformer',
        help='The landmaek predictor to use.'
    )

    # Ckpt
    parser.add_argument(
        '--load_ckpt', default=None,
        help='The path to load model checkpoint weights from.'
    )
    parser.add_argument(
        '--no_save_ckpts', action='store_true',
        help='If set, don\'t save checkpoints during training.'
    )
    parser.add_argument(
        '--best_metric', default='GaussianNLL',
        help='If this metric improves, create a checkpoint '
        '(when --save_best is set).'
    )
    parser.add_argument(
        '--higher_is_better', action='store_true',
        help='If set, the metric set with --best_metric is better when it '
        'inreases.'
    )
    parser.add_argument(
        '--ckpts_path', default='./ckpts',
        help='The directory to save checkpoints.'
    )

    # K-Fold args
    parser.add_argument(
        '--k_fold_seed', default=15,
        help='Seed for the dataset shuffle used to create the K folds.',
        type=int
    )
    parser.add_argument(
        '--k_fold_num_folds', default=20,
        help='The number of folds to use.',
        type=int
    )
    parser.add_argument(
        '--k_fold_val_fold', default=0,
        help='The index of the validation fold. '
        'If None, all folds are used for training.',
        type=int
    )

    # Dataset
    parser.add_argument(
        '--data_path_300w', default='/apollo/datasets/300W',
        help='Path to the 300W dataset.',
    )
    parser.add_argument(
        '--data_path_fitymi', default='/apollo/datasets/FITYMI',
        help='Path to the Fake-It-Till-You-Make-It dataset.',
    )
    parser.add_argument(
        '--data_path_facescape', default='/apollo/datasets/FaceScape_512',
        help='Path to the FaceScape dataset.',
    )

    # Data augmentations
    parser.add_argument(
        '--input_size', default=224,
        help='Input size of the feature extractor'
    )
    parser.add_argument(
        '--rrc_scale', default=(0.08, 1.0),
        help='Random resized crop scale'
    )
    parser.add_argument(
        '--rrc_ratio', default=(3/4, 4/3),
        help='Random resized crop aspect ratio'
    )
    parser.add_argument(
        '--random_angle', default=0,
        help='Random angle'
    )
    parser.add_argument(
        '--random_brightness', default=0.1,
        help='Brightness jitter'
    )
    parser.add_argument(
        '--random_contrast', default=0.1,
        help='Contrast jitter'
    )
    parser.add_argument(
        '--random_saturation', default=0.1,
        help='Saturation jitter'
    )
    parser.add_argument(
        '--norm_mean', default=[0.5, 0.5, 0.5],
        help='Image normalization mean'
    )
    parser.add_argument(
        '--norm_std', default=[0.2, 0.2, 0.2],
        help='Image normalization std'
    )

    # Dataloader args
    parser.add_argument('--batch_size', default=64,
                        help='The training batch size.', type=int)
    parser.add_argument('--val_batch_size', default=64,
                        help='The validation batch size.', type=int)
    parser.add_argument(
        '--num_workers', default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Optimizer args
    parser.add_argument('--lr', default=0.0001, help='The learning rate.',
                        type=float)
    parser.add_argument('--beta1', default=0.95, help='The beta1 of AdamW.',
                        type=float)
    parser.add_argument('--beta2', default=0.999, help='The beta2 of AdamW.',
                        type=float)
    parser.add_argument('--weight_decay', default=0,
                        help='The weight decay.',
                        type=float)
    parser.add_argument('--lr_warmup_steps', default=100, help='The number of '
                        'learning rate warmup steps.',
                        type=int)

    # Train args
    parser.add_argument(
        '--num_epochs', default=30,
        help='The number of epochs to train.',
        type=int
    )
    parser.add_argument(
        '--val_every', default=1000,
        help='Run a validation epoch after this number of iterations.',
        type=int,
    )

    # Log args
    parser.add_argument(
        '--wandb_entity', help='Weights and Biases entity.'
    )
    parser.add_argument(
        '--wandb_project', help='Weights and Biases project.'
    )

    # Device arg
    parser.add_argument('--device', default='cuda',
                        help='The device (cuda/cpu) to use.')

    args = parser.parse_args()

    args_dict = vars(args)
    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               config=args_dict)

    del args_dict['wandb_entity']
    del args_dict['wandb_project']
    run_training(**vars(args))
