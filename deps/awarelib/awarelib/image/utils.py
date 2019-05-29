import subprocess
import tempfile
from PIL import Image



def show_numpy_img_pil(array):
    img = Image.fromarray(array, 'RGB')
    img.show()


def show_numpy_img_sxiv(array):
    img = Image.fromarray(array, 'RGB')
    tmp_path = tempfile.mktemp(suffix='img.jpg')
    print(tmp_path)
    img.save(tmp_path)
    viewer = subprocess.Popen(['sxiv', tmp_path])
    viewer.wait()


def show_numpy_img_plot(array):
    from matplotlib import pyplot as plt
    plt.imshow(array, interpolation='nearest')
    plt.show()
