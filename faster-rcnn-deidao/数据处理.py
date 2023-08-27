from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
pimg='1.jpg'
with open(pimg, 'rb') as f:
    #这样读就是二进制的
    f = f.read()
#这句 就是补全数据的
f=f+B'\xff'+B'\xd9'

im = Image.open(BytesIO(f))
if im.mode != "RGB":
    im = im.convert('RGB')
imr = im.resize((256, 256), resample=Image.BILINEAR)
imr.show()
