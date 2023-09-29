from pathlib import Path
from torch.utils.data import DataLoader

from continuous_landmarks.dataset.transforms import (
    Compose, Align, RandomResizedCrop,
    ToTensor,
)
from continuous_landmarks.dataset import face300w
from continuous_landmarks.model import FeatureExtractor, LandmarkPredictor,\
    PositionEncoder


pos_encoder = PositionEncoder()
feat_extractor = FeatureExtractor('ConvNeXt')
lm_predictor = LandmarkPredictor(
    query_size=pos_encoder.encoding_size,
    feature_size=feat_extractor.feature_size,
    model_name='Transformer',
)


ds = face300w.Face300WDataset(
    Path('../data/300W/'),
    Compose([
        Align(face300w.get_eyes_mouth),
        RandomResizedCrop(224),
        ToTensor(),
    ])
)
dl = DataLoader(ds, batch_size=10, shuffle=True)


img_batch, lm_batch = next(iter(dl))
canon_lms = dl.dataset.canonical[None, :, :].expand((dl.batch_size, -1, -1))
B, N, _ = lm_batch.shape
query_sequence = pos_encoder(canon_lms.flatten(end_dim=1)).unflatten(0, (B, N))
feature = feat_extractor(img_batch)
lm_pred = lm_predictor(query_sequence, feature)
