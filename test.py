import os
import sys
import numpy as np
import torch
import argparse
import logging
import torch.utils
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
from multi_read_data import DataLoader
from thop import profile



root_dir = os.path.abspath('../')
sys.path.append(root_dir)

parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--data_path_test_low', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,
                    default='./results/',
                    help='location of the data corpus')
parser.add_argument('--model_test', type=str,
                    default='./model',
                    help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save
os.makedirs(save_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
mertic = logging.FileHandler(os.path.join(args.save, 'log.txt'))
mertic.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(mertic)

logging.info("train file name = %s", os.path.split(__file__))
TestDataset = DataLoader(img_dir=args.data_path_test_low,task='test')
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im

def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_model_flops(model, input_tensor):
    flops, _ = profile(model, inputs=(input_tensor,))
    flops_in_gigaflops = flops / 1e9  # Convert FLOPs to gigaflops (G)
    return flops_in_gigaflops

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model_test)
    model = model.cuda()
    model.eval()
    # Calculate model size
    total_params = calculate_model_parameters(model)
    print("Total number of parameters: ", total_params)
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for _, (input,  img_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            input_name = img_name[0].split('/')[-1].split('.')[0]
            enhance,output = model(input)
            input_name = '%s' % (input_name)
            enhance=save_images(enhance)
            output = save_images(output)
            os.makedirs(args.save + '/result', exist_ok=True)
            Image.fromarray(output).save(args.save + '/result/' +input_name + '_denoise' + '.png', 'PNG')
            Image.fromarray(enhance).save(args.save + '/result/'+ input_name + '_enhance'  + '.png', 'PNG')
    torch.set_grad_enabled(True)


if __name__ == '__main__':
    main()
