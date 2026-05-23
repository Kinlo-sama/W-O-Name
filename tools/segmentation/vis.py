from pathlib import Path
import cv2
import numpy as np
import sys

DATA_DIR = Path(sys.argv[1])

IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"

image_paths = sorted(list(IMAGES_DIR.glob("*.jpg")))

for image_path in image_paths:

    stem = image_path.stem
    mask_path = MASKS_DIR / f"{stem}.png"

    if not mask_path.exists():
        continue

    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


    mask_raw = mask.copy()


    mask_vis = mask * 80  # scale for visibility
    mask_vis = np.clip(mask_vis, 0, 255).astype(np.uint8)


    overlay = img.copy()

    colored_mask = np.zeros_like(img)

    colored_mask[mask > 0] = [0, 0, 255]  # red foreground

    overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

    img = cv2.resize(img, (512, 512))
    mask_raw = cv2.resize(mask_raw, (512, 512))
    mask_vis = cv2.resize(mask_vis, (512, 512))
    overlay = cv2.resize(overlay, (512, 512))

    top = np.hstack([img, overlay])
    bottom = np.hstack([
        cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    ])

    final = np.vstack([top, bottom])

    cv2.imshow("Image | Overlay | Mask raw | Mask vis", final)

    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()