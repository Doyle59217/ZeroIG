import os
import sys
import time
import glob
import numpy as np
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import *
from multi_read_data import DataLoader


parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=2001, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--save', type=str, default='./EXP/', help='location of the data corpus')
parser.add_argument('--model_pretrain', type=str,default='',help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)



    model =Network()
    utils.save(model, os.path.join(args.save, 'initial_weights.pt'))
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)
    train_low_data_names = './data/1'
    TrainDataset = DataLoader(img_dir=train_low_data_names, task='train')

    test_low_data_names = './data/1'
    TestDataset = DataLoader(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device='cuda'))
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device='cuda'))

    total_step = 0
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for idx, (input, img_name) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()
            optimizer.zero_grad()
            optimizer.param_groups[0]['capturable'] = True
            loss = model._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            logging.info('train-epoch %03d %03d %f', epoch, idx, loss)
        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        if epoch % 50 == 0 and total_step != 0:
            model.eval()
            with torch.no_grad():
                for idx, (input, img_name) in enumerate(test_queue):
                    input = Variable(input, volatile=True).cuda()
                    image_name = img_name[0].split('/')[-1].split('.')[0]
                    L_pred1,L_pred2,L2,s2,s21,s22,H2,H11,H12,H13,s13,H14,s14,H3,s3,H3_pred,H4_pred,L_pred1_L_pred2_diff,H13_H14_diff,H2_blur,H3_blur= model(input)
                    input_name = '%s' % (image_name)
                    H3 = save_images(H3)
                    H2= save_images(H2)
                    os.makedirs(args.save + '/result/denoise/', exist_ok=True)
                    os.makedirs(args.save + '/result/enhance/', exist_ok=True)
                    Image.fromarray(H3).save(args.save + '/result/denoise/' + input_name+'_denoise_'+str(epoch)+'.png', 'PNG')
                    Image.fromarray(H2).save(args.save + '/result/enhance/' +input_name+'_enhance_'+str(epoch)+'.png', 'PNG')


if __name__ == '__main__':
    main()