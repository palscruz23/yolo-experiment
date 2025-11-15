import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import os


EXPORT_DIR = "open-image/dataset"

if not os.path.exists('open-image/dataset'):
    os.mkdir('open-image/dataset')
export_path = Path(EXPORT_DIR)

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split = "train",
    label_types = ["detections"],
    classes = ["Human beard",
            "Human ear",
            "Human eye",
            "Human face",
            "Human hair",
            "Human mouth",
            "Human nose"],
    max_samples = 1000
)

dataset.export(
    export_dir = str(export_path.absolute()),
    dataset_type = fo.types.YOLOv5Dataset,
    label_field = "detections"
)

# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split = "val",
#     label_types = ["detections"],
#     classes = ["Human beard",
#             "Human ear",
#             "Human eye",
#             "Human face",
#             "Human hair",
#             "Human mouth",
#             "Human nose"],
#     max_samples = 200
# )

# dataset.export(
#     export_dir = "oiv7_yolo_subset",
#     dataset_type = fo.types.YOLOv5Dataset,
#     label_field = "detections"
# )