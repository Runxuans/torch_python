from cv2 import cv2


def change(img, max_height, max_width):
    h = img.shape[0]
    w = img.shape[1]
    scale = 1
    if h > max_height:
        scale = max_height / h
        if w*scale>max_width:
            scale = max_width/w
    elif w > max_width:
        scale = max_width/w
        if h*scale > max_height:
            scale = max_height/h

    imgResize = cv2.resize(img,dsize=None, fx=scale, fy=scale)
    return imgResize



