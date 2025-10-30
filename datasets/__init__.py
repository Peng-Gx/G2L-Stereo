from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDataset
from .eth3d_dataset import ETH3DDataset
from .middlebury_dataset import MiddleburyDataset
from .booster_dataset import BoosterDataset
from .kitti_raw_dataset import KITTIrawDataset
from .drivingstereo_dataset import DrivingStereoDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "eth3d": ETH3DDataset,
    "middlebury": MiddleburyDataset,
    "booster": BoosterDataset,
    "kitti_raw": KITTIrawDataset,
    "drivingstereo": DrivingStereoDataset
}
