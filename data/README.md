# Dataset format: image_folder

Place your images in class-named subfolders under `train/` and `val/`.

```
data/
  train/
    0/ digit_0001.png
    1/ digit_0002.png
    ...
    9/ digit_9999.png
  val/
    0/ ...
    1/ ...
    ...
    9/ ...
```

Rules:

- There must be exactly 10 class folders named `0` through `9`.
- Images can be PNG/JPG. MNIST images are grayscale 28x28; they will be resized to the configured `image_size` and converted to RGB when needed.
- Every image inside a folder is assumed to have that folder's label.

## Getting MNIST as image folders

If you only have the raw MNIST dataset, you can export it to this layout with a short script using `torchvision.datasets.MNIST` and saving each image into `train/<label>/` and `val/<label>/`. Any tool that produces the structure above will work.
