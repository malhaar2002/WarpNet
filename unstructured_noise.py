from PIL import Image
import numpy as np
import random
import cv2
from io import BytesIO

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def synthesize_salt_pepper(image,amount,salt_vs_pepper):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    out = img_pil.copy()
    p = amount
    q = salt_vs_pepper
    flipped = np.random.choice([True, False], size=img_pil.shape,
                               p=[p, 1 - p])
    salted = np.random.choice([True, False], size=img_pil.shape,
                              p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 1
    out[flipped & peppered] = 0.
    noisy = np.clip(out, 0, 1).astype(np.float32)


    return np_to_pil(noisy)

def synthesize_gaussian(image,std_l,std_r):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)

def synthesize_speckle(image,std_l,std_r):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return np_to_pil(noisy)


def synthesize_low_resolution(img):
    w,h=img.size
    print(w, h)
    new_w=random.randint(int(w/2),w)
    new_h=random.randint(int(h/2),h)

    img=img.resize((new_w,new_h),Image.BICUBIC)

    if random.uniform(0,1)<0.5:
        img=img.resize((w,h),Image.NEAREST)
    else:
        img = img.resize((w, h), Image.BILINEAR)

    return img


def convertToJpeg(im,quality):
    with BytesIO() as f:
        im.save(f, format='JPEG',quality=quality)
        f.seek(0)
        return Image.open(f).convert('RGB')


def blur_image(img):
    x=np.array(img)
    kernel_size_candidate=[(3,3),(5,5),(7,7)]
    kernel_size=random.sample(kernel_size_candidate,1)[0]
    std=random.uniform(1.,5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,std)

    return Image.fromarray(blur.astype(np.uint8))

def online_add_degradation(img):

    task_id=np.random.permutation(4)

    for x in task_id:
        if x==0 and random.uniform(0,1)<0.7:
            img = blur_image(img)
        if x==1 and random.uniform(0,1)<0.7:
            flag = random.choice([1, 2, 3])
            if flag == 1:
                img = synthesize_gaussian(img, 5, 50)
            if flag == 2:
                img = synthesize_speckle(img, 5, 50)
            if flag == 3:
                img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
        if x==2 and random.uniform(0,1)<0.7:
            img=synthesize_low_resolution(img)

        if x==3 and random.uniform(0,1)<0.7:
            img=convertToJpeg(img,random.randint(40,100))

    return img


def irregular_hole_synthesize(img,mask):

    img_np=np.array(img).astype('uint8')
    mask_np=np.array(mask).astype('uint8')
    mask_np=mask_np/255
    img_new=img_np*(1-mask_np)+mask_np*255


    hole_img=Image.fromarray(img_new.astype('uint8')).convert("RGB")

    return hole_img,mask.convert("L")

def zero_mask(size):
    x=np.zeros((size,size,3)).astype('uint8')
    mask=Image.fromarray(x).convert("RGB")
    return mask