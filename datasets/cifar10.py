import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    for split in ["test"]: # 分别处理训练和测试集 ["train", "test"]
        out_dir = f"cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        # with tempfile.TemporaryDirectory() as tmp_dir: # tempfile.TemporaryDirectory():创建临时目录,临时目录路径赋值给tmp_dir
        #     dataset = torchvision.datasets.CIFAR10(
        #         root=tmp_dir, train=split == "train", download=True
        #     )
        tmp_dir = "/home/allenyljiang/Desktop/d2l-pytorch/data/"
        dataset = torchvision.datasets.CIFAR10(root=tmp_dir,train=False,download=False)

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
