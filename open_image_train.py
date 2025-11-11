import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split = "validation",
    label_types = ["detections"],
    classes = ["Car", "Person"],
    max_samples = 1000
)

dataset.export(
    export_dir = "oiv7_yolo_subset",
    dataset_type = fo.types.YOLOv5Dataset,
    label_field = "detections"
)