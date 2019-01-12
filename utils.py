from PIL import Image
import numpy as np
import pickle


def resize_image(in_image, new_width=224, new_height=224, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    # 将一个Image文件给修改成224 * 224的图片大小
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def pil_to_nparray(pil_image):
    # 将Image加载后转换成float32格式的tensor
    pil_image.load()
    return np.asarray(pil_image, dtype='float32')

def load_with_pickle(file_path):
    file = open(file_path, 'rb')
    images, labels = pickle.load(file)
    return images, labels