import cv2
import numpy as np

def add_text(image: np.ndarray, text: str, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    # 在图像上绘制文本
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_text_list(image: np.ndarray, text_list: list, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    # 在图像上绘制文本
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_rectangle(image: np.ndarray, top_left: tuple, bottom_right: tuple, color=(0, 255, 0), thickness=2):
    # 在图像上绘制矩形框
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def add_resized_image(base_image: np.ndarray, overlay_image: np.ndarray, position: tuple, size: tuple):
    # 调整叠加图像的大小
    resized_overlay = cv2.resize(overlay_image, size)

    # 获取叠加图像的高度和宽度
    h, w = resized_overlay.shape[:2]

    # 计算叠加图像在基础图像中的位置
    x, y = position

    # 确保叠加图像不会超出基础图像边界
    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError("Overlay image goes out of the bounds of the base image.")

    # 将调整大小后的图像放置在基础图像上
    base_image[y:y+h, x:x+w] = resized_overlay
    return base_image

def crop_around_point(image: np.ndarray, point: tuple, size: tuple):
    """
    从给定的图像中截取以指定点为中心的矩形区域。
    
    :param image: 输入图像，形状为 (height, width, channels)
    :param point: 关注点的坐标 (x, y)，其中 x 是横轴方向，y 是纵轴方向
    :param size: 截取区域的尺寸 (width, height)
    :return: 截取后的图像
    """
    # 获取图像的高度和宽度
    img_height, img_width = image.shape[:2]
    
    # 获取截取区域的尺寸
    crop_width, crop_height = size
    
    # 计算截取区域的边界
    left = max(point[0] - crop_width // 2, 0)
    top = max(point[1] - crop_height // 2, 0)
    right = min(point[0] + (crop_width - crop_width // 2), img_width)
    bottom = min(point[1] + (crop_height - crop_height // 2), img_height)
    
    # 确保截取区域不超出图像边界
    if right - left < crop_width:
        if left == 0:
            right = left + crop_width
        else:
            left = right - crop_width
    if bottom - top < crop_height:
        if top == 0:
            bottom = top + crop_height
        else:
            top = bottom - crop_height
    
    # 截取图像
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

# 示例
# image = np.zeros((400, 600, 3), dtype=np.uint8)  # 创建一个空白图像
# image_with_text = add_text(image, "Hello, World!", (50, 200), font_scale=2)
# cv2.imshow('Image with Text', image_with_text)
# cv2.waitKey(0)
# cv2.destroyAllWindows()