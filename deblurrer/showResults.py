import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if len(sys.argv) < 2:
    print("Usage: python showResults.py [filename]")
    sys.exit(1)

filename = sys.argv[1]

sharp_path = os.path.join('testing_sharp', filename)
blur_path = os.path.join('testing_blur', filename)
result_path = os.path.join('testing_results', filename)

sharp_img = mpimg.imread(sharp_path)
blur_img = mpimg.imread(blur_path)
result_img = mpimg.imread(result_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sharp_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(blur_img)
axes[1].set_title('Blurred Image')
axes[1].axis('off')

axes[2].imshow(result_img)
axes[2].set_title('Deblurred Image')
axes[2].axis('off')

plt.show()