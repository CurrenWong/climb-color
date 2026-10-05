import cv2, numpy as np, glob, os

files = sorted(glob.glob('result_edgesam_*.jpg'),
               key=lambda f: int(f.split('_')[-1].split('.')[0]))
imgs = [cv2.imread(f) for f in files if os.path.getsize(f) > 0]
h = min(i.shape[0] for i in imgs)
imgs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
cols, rows = 5, 3
cell_w = max(i.shape[1] for i in imgs)
canvas = np.zeros((rows * (h + 12), cols * cell_w, 3), np.uint8)
for r in range(rows):
    for c in range(cols):
        idx = r * cols + c
        if idx < len(imgs):
            y0 = r * (h + 12)
            x0 = c * cell_w
            canvas[y0:y0 + h, x0:x0 + imgs[idx].shape[1]] = imgs[idx]
cv2.imwrite('_check_15grid.jpg', canvas)
print('grid saved', canvas.shape, len(files), 'files')
