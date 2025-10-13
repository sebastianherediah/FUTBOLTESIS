"""Script de entrenamiento de RF-DETR con Roboflow según el flujo proporcionado."""

from roboflow import Roboflow
from rfdetr import RFDETRBase
from PIL import Image

rf = Roboflow(api_key="dcZKRj2xqwPU94CoAUcS")
project = rf.workspace("sebas-xxi8x").project("dataset-millonarios")
version = project.version(4)
dataset = version.download("coco")

model = RFDETRBase()
model.train(
    dataset_dir=dataset.location,
    epochs=15,
    batch_size=4,
    grad_accum_steps=1,
    lr=1e-4,
)

Image.open("/content/output/metrics_plot.png")

model = RFDETRBase(pretrain_weights="/content/output/checkpoint_best_total.pth")
model.export()  # produces an ONNX file under your output directory
