import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image



def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1,output2

def gauss_cdf(x):
    return 0.5*(1+torch.erf(x/torch.sqrt(torch.tensor(2.))))

def gauss_kernel(kernlen=21,nsig=3,channels=1):
    interval=(2*nsig+1.)/(kernlen)
    x=torch.linspace(-nsig-interval/2.,nsig+interval/2.,kernlen+1,).cuda()
    #kern1d=torch.diff(torch.erf(x/math.sqrt(2.0)))/2.0
    kern1d=torch.diff(gauss_cdf(x))
    kernel_raw=torch.sqrt(torch.outer(kern1d,kern1d))
    kernel=kernel_raw/torch.sum(kernel_raw)
    #out_filter=kernel.unsqueeze(2).unsqueeze(3).repeat(1,1,channels,1)
    out_filter=kernel.view(1,1,kernlen,kernlen)
    out_filter = out_filter.repeat(channels,1,1,1)
    return  out_filter

class LocalMean(torch.nn.Module):
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    def forward(self, image):
        image = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))

def blur(x):
    device = x.device
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1)).to(device)
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')
    return torch.nn.functional .conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))

def padr_tensor(img):
    pad=2
    pad_mod=torch.nn.ConstantPad2d(pad,0)
    img_pad=pad_mod(img)
    return img_pad

def calculate_local_variance(train_noisy):
    b,c,w,h=train_noisy.shape
    avg_pool = torch.nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
    noisy_avg= avg_pool(train_noisy)
    noisy_avg_pad=padr_tensor(noisy_avg)
    train_noisy=padr_tensor(train_noisy)
    unfolded_noisy_avg=noisy_avg_pad.unfold(2,5,1).unfold(3,5,1)
    unfolded_noisy=train_noisy.unfold(2,5,1).unfold(3,5,1)
    unfolded_noisy_avg=unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0],-1,5,5)
    unfolded_noisy=unfolded_noisy.reshape(unfolded_noisy.shape[0],-1,5,5)
    noisy_diff_squared=(unfolded_noisy-unfolded_noisy_avg)**2
    noisy_var=torch.mean(noisy_diff_squared,dim=(2,3))
    noisy_var=noisy_var.view(b,c,w,h)
    return noisy_var

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'),exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def show_pic(pic, name,path):
    pic_num = len(pic)
    for i in range(pic_num):
        img = pic[i]
        image_numpy = img[0].cpu().float().numpy()
        if image_numpy.shape[0]==3:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
            im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im)
        elif image_numpy.shape[0]==1:
            im = Image.fromarray(np.clip(image_numpy[0] * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im,plt.cm.gray)
    plt.savefig(path)



