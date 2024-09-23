import numpy as np
from tensorflow import convert_to_tensor
from PIL import Image

def load_img(image, max_dim=512):
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img = np.array(image).astype(np.float32)

    # Resize maintaining aspect ratio
    long_dim = max(image.size)
    scale = max_dim / long_dim
    new_shape = (np.array(image.size) * scale).astype(np.int32)

    # Convert new_shape to a tuple for the resize function
    new_shape_tuple = tuple(new_shape)

    img = np.array(image.resize(new_shape_tuple)).astype(np.float32)[np.newaxis, ...] / 255.0
    
    # Ensure the shape is (1, height, width, 3) -> no alpha channel (4th channel)
    if img.shape[-1] == 4:
        img = img[..., :3]  # Drop the alpha channel
    
    return convert_to_tensor(img)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)