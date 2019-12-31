import torch, sys, cv2, os
import numpy as np
import argparse
sys.path.insert(1, 'pytorch_deep_image_matting/core')
import net
from torchvision import transforms
from PIL import Image
import imageio

def model_dim_fn(cuda):
    stage = 1
    resume = "pytorch_deep_image_matting/model/stage1_sad_54.4.pth"
    model = net.VGG16(stage)
    if cuda:
        ckpt = torch.load(resume)
    else :
        ckpt = torch.load(resume, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    
    if cuda:
        model = model.cuda()
    
    return model

def inference_once(model, scale_img, scale_trimap, cuda):
    size_h = 320
    size_w = 320
    stage = 1
       
    #if aligned:
    #    assert(scale_img.shape[0] == size_h)
    #    assert(scale_img.shape[1] == size_w)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)
    # first, 0-255 to 0-1
    # second, x-mean/std and HWC to CHW
    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    scale_grad = compute_gradient(scale_img)
    #tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])
    tensor_grad = torch.from_numpy(scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :])

    if cuda:
        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()
        tensor_grad = tensor_grad.cuda()
    
    input_t = torch.cat((tensor_img, tensor_trimap / 255.), 1)

    # forward
    if stage <= 1:
        # stage 1
        pred_mattes, _ = model(input_t)
    else:
        # stage 2, 3
        _, pred_mattes = model(input_t)
        
    pred_mattes = pred_mattes.data
    if cuda:
        pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]

    return pred_mattes

def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad

def inference_img_whole(max_size, model, img, trimap, cuda): #args
    h, w, c = img.shape
    #new_h = min(args.max_size, h - (h % 32))
    #new_w = min(args.max_size, w - (w % 32))
    new_h = min(max_size, h - (h % 32))
    new_w = min(max_size, w - (w % 32))
    
    # resize for network input, to Tensor
    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(model, scale_img, scale_trimap, cuda)#args

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(origin_pred_mattes.shape == trimap.shape)
    return origin_pred_mattes

def deep_image_matting_final(model, image, trimap, cuda):
    torch.cuda.empty_cache()
    with torch.no_grad():
        pred_mattes = inference_img_whole(1600, model, image, trimap, cuda)
    return pred_mattes

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im

def matting_result(pic_input, tri_input, model, cuda, website = False):
    if website:
        img = pic_input 
    else :
        img = imageio.imread(pic_input)[:, :, :3]
    trimap = tri_input
    if len(trimap.shape)>2:
        trimap = trimap[:, :, 0]
    alpha = deep_image_matting_final(model, img, trimap, cuda)  
    alpha[trimap == 0] = 0.0
    alpha[trimap == 255] = 1.0
    h, w = img.shape[:2]
    new_bg = np.array(np.full((h,w,3), 255), dtype='uint8')
    im = composite4(img, new_bg, alpha, w, h)
    return Image.fromarray(im)