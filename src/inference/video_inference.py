from roboflow import Roboflow
rf = Roboflow(api_key="dcZKRj2xqwPU94CoAUcS")
project = rf.workspace("sebas-xxi8x").project("dataset-millonarios")
version = project.version(4)
dataset = version.download("coco")
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=dataset.location, epochs=15, batch_size=4, grad_accum_steps=1, lr=1e-4)

from PIL import Image

Image.open("/content/output/metrics_plot.png")

model = RFDETRBase(pretrain_weights="/content/output/checkpoint_best_total.pth")
model.export()  # produces an ONNX file under your output directory
