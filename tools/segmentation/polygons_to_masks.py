from pathlib import Path
import cv2 
import numpy as np
import sys

DATA_DIR = Path(
    sys.argv[1]
)

ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

MASKS_DIR.mkdir(
    exist_ok=True,
    parents=True
)

image_paths = list(
    IMAGES_DIR.glob("*.jpg")
)

for image_path in image_paths:
    stem = image_path.stem
    annotation_path = ANNOTATIONS_DIR / f"{stem}.txt"

    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    with open(annotation_path) as f:
        lines = f.readlines()
    
    for line in lines:
        values = line.strip().split()
    
        classid = int(values[0])

        coords = list(
            map(float, values[1:])
        )
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x,y])

        points = np.array(
            points,
            dtype=np.int32
        )

        cv2.fillPoly(
            mask, 
            [points],
            color = classid + 1
        )

    mask_path = (
        MASKS_DIR / 
        f"{stem}.png"
    )
    cv2.imwrite(
        str(mask_path),
        mask
    )