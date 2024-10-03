import time
import copy
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import transforms
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from resnet_wide import WRN_40_4, WRN_16_4
from resnet_20 import resnet20
from functorch import make_functional_with_buffers, combine_state_for_ensemble
from functorch import vmap, grad
from copy import deepcopy
import argparse
from functools import partial
from accounting_analysis import get_std
from torch_ema import ema
def get_per_grad(inputs, targets, worker_model_func, worker_param_func, worker_buffers_func,chain_len,lr_0):

    def compute_loss(model_para, buffers, inputs, targets):
        loss_metric = nn.CrossEntropyLoss()
        # print(f'inputs shape: {inputs.shape}')
        predictions = worker_model_func(model_para, buffers, inputs)
        # print(f'predictions shape: {predictions.shape}, targets shape: {targets.shape}')
        ''' only compute the loss of the first(private) sample '''
        loss = loss_metric(predictions, targets.flatten())  # * inputs.shape[0]
        return loss

    def self_aug_per_grad(model_para, buffers, inputs, targets,chain_len,lr_0):
        init_model = [p.clone() for p in model_para]
        # running_model = [torch.clone(p) for p in model_para]
        momemtum = [0 for _ in range(len(model_para))]
        chain_len = chain_len  # self.arg_setup.chain_len
        beta = 0.5  # self.arg_setup.forward_beta
        # lr_0 = 1
        lr_0 = lr_0
        for _ in range(chain_len):
            per_grad = grad(compute_loss)(model_para, buffers, inputs, targets)
            # for p in per_grad:
            #     print(p.shape)
            momemtum = [beta * m + (1-beta)*g for m, g in zip(momemtum, per_grad)]
            for p_worker, p_momemtum in zip(model_para, momemtum):
                # print(p_worker.data.shape, p_momemtum.shape)
                p_worker.add_(- lr_0 * p_momemtum)
        per_grad = [i - p for p, i in zip(model_para, init_model)]
        return list(per_grad)

    self_aug_per_grad_partial = partial(self_aug_per_grad, chain_len=chain_len, lr_0=lr_0)
    per_grad = vmap(self_aug_per_grad_partial, in_dims=(0, 0, 0, 0))(worker_param_func, worker_buffers_func,inputs,targets)

    return per_grad

def flatten_to_rows(leading_dim, iterator):
    return torch.cat([p.reshape(leading_dim, -1) for p in iterator], dim = 1)
def sampling_noise_summary(index, per_grad):
    grad_flatten = flatten_to_rows(per_grad[0].shape[0], per_grad)
    grad_flatten_mean = torch.mean(grad_flatten, dim=0, keepdim=True)
    center_around_mean = grad_flatten - grad_flatten_mean
    grad_norm = torch.norm(grad_flatten, dim=1)
    mean_of_grad_norm = grad_norm.mean()
    clipnorm = float(mean_of_grad_norm)
    grad_norm_0 = grad_norm - clipnorm
    grad_norm_0[grad_norm_0 < 0] = 0
    return clipnorm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=80,help='epoch')
    parser.add_argument('--batch_size', type=int, default=250, help='batch_size')
    parser.add_argument('--learning_rate', type=int, default=2, help='learning_rate')
    parser.add_argument('--clip_threshold', type=int, default=1, help='clip_threshold')
    #parser.add_argument('--noise_multiplier', type=float, default=1.226, help='noise_multiplier')
    parser.add_argument('--self_aug', type=int, default=8, help='self_aug')
    parser.add_argument('--accumulation_steps', type=int, default=10, help='accumulation_steps:串行训练，每迭代accumulation_steps次更新网络')
    parser.add_argument('--chain_len', type=int, default=1, help='chain_length')
    parser.add_argument('--lr_0', type=float, default=1, help='lr_0')
    parser.add_argument('--epsilon', type=float, default=6, help='lr_0')
    parser.add_argument('--model', type=str, default='wid_res_40', help='[resnet20,wid_res16,wid_res_40]')
    parser.add_argument('--True_batch_size', help='batch_size*accumulation_steps')
    arg = parser.parse_args()
    
    eps = arg.epsilon
    print('epsilon', eps)
    print('learning rate', 2) 

    device = torch.device("cuda")
    total_epochs = arg.epoch
    batch_size = arg.batch_size
    DATASET_PATH = './data'
    torch.backends.cudnn.benchmark = True

    # 准备数据
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),

        ]),
        'valid': transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_transforms_input = T.Compose([
        # T.ToPILImage(),
        # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 均值，标准差
        T.RandomCrop(size=(32, 32), padding=4),
        T.RandomHorizontalFlip(),
        # T.RandomRotation(degrees=(-10, 10),),
        # T.ToTensor(),
    ])
    image_datasets = {
        x: CIFAR10(DATASET_PATH, train=True if x == 'train' else False,
                   transform=data_transforms[x], download=True) for x in ['train', 'valid']}
    dataloaders: dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False
        ) for x in ['train', 'valid']
    }
    if arg.model == 'wid_res16':
        model = WRN_16_4()
        model.to(device)
        print('model:WRN_16_4')
    elif arg.model=='resnet20':
        model = resnet20(10)
        model.to(device)
        print('model:resnet20')
    elif arg.model == 'wid_res40':
        model = WRN_40_4()
        model.to(device)
        print('model:WRN_40_4')
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    counter = 0
    total_step = {
        'train': 0, 'valid': 0
    }
    # 记录开始时间
    since = time.time()
    clip_threshold = arg.clip_threshold
    noise_multiplier= 0.85*get_std(q=arg.batch_size * arg.accumulation_steps / 50000,EPOCH=arg.epoch, epsilon=arg.epsilon, delta=1e-5, verbose=True)
    opt = optim.SGD(model.parameters(), lr=arg.learning_rate, momentum=0.75, weight_decay=1e-4)
    ema = ema.ExponentialMovingAverage(model.parameters(), decay=0.999)

    for epoch in range(total_epochs):
        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('-' * 10)
        print()
        for phase in ['train']:
            model.train()  # 训练
            running_loss = 0.0
            running_corrects = 0
            grad_list = []
            accumulation_steps = 0
            aug_num = arg.self_aug
            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # worker_model_func, worker_param_func, worker_buffers_func = make_functional_with_buffers(deepcopy(model), disable_autograd_tracking=True)
                models = [deepcopy(model) for _ in range(batch_size)]
                worker_model_func, worker_param_func, worker_buffers_func = combine_state_for_ensemble(models)
                per_grad_list = []
                for i in range(aug_num):
                    for i in range(batch_size):
                        inputs[i,:,:,:] = data_transforms_input(inputs[i,:,:,:])
                    new_inputs = torch.stack(torch.split(inputs, 1, dim=0))
                    new_targets = torch.stack(torch.split(labels, 1, dim=0))
                    per_grad = get_per_grad(new_inputs, new_targets, worker_model_func, worker_param_func,worker_buffers_func, arg.chain_len, arg.lr_0)
                    if per_grad_list == []:
                        per_grad_list = per_grad
                    else:
                        for num in range(len(per_grad)):
                            per_grad_list[num] = per_grad_list[num] + per_grad[num]
                for num in range(len(per_grad_list)):
                    per_grad_list[num] = per_grad_list[num] / (aug_num + 1)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                model_10_grad_sample_list = per_grad_list
                #clip_threshold = sampling_noise_summary(0, model_10_grad_sample_list)
                per_sample_clip_factor = []
                for i in range(len(labels)):   # 计算clip_factor
                    for j in range(len(model_10_grad_sample_list)):
                        model_j = model_10_grad_sample_list[j]
                        model_j_i = model_j[i]
                        if j == 0:
                            sample_sum_grad = torch.flatten(model_j_i)
                        elif j > 0:
                            sample_sum_grad = torch.cat((sample_sum_grad, torch.flatten(model_j_i)), dim=0)
                    sample_sum_grad_norm = torch.norm(sample_sum_grad)
                    sample_clip_factor = (clip_threshold / (sample_sum_grad_norm + 1e-6)).clamp(max=1.0)
                    per_sample_clip_factor.append(sample_clip_factor)

                num_network = 0
                for _ in model.named_parameters():
                    model_10_grad_sample = model_10_grad_sample_list[num_network]
                    for i in range(batch_size):
                        model_10_grad_sample[i] = model_10_grad_sample[i] * per_sample_clip_factor[i]  # clip
                    p_grad = torch.sum(model_10_grad_sample, dim=0)
                    # p_grad += max_norm * noise_multiplier * torch.randn_like(p_grad)
                    p_grad /= batch_size

                    if accumulation_steps == 0:
                        grad_list.append(p_grad)
                    else:
                        grad_list[num_network] = grad_list[num_network] + p_grad
                    num_network = num_network + 1

                accumulation_steps = accumulation_steps + 1
                if accumulation_steps == arg.accumulation_steps:
                    num_network = 0
                    for p in model.parameters():
                        p.grad = grad_list[num_network] / accumulation_steps
                        p.grad += (clip_threshold * noise_multiplier * torch.randn_like(p.grad)) / (batch_size * accumulation_steps)  # add noise
                        num_network = num_network + 1
                    opt.step()  # 优化权重

                    ema.update()  # Parameter averaging (exponential moving average)

                    accumulation_steps = 0
                    grad_list = []
                opt.zero_grad()
                running_loss += 0
                running_corrects += (preds == labels).sum()  # 计算预测正确总个数
                total_step[phase] += 1
            epoch_loss = running_loss / len(dataloaders['train'].sampler)  # 当前轮的总体平均损失值
            epoch_acc = float(running_corrects) / len(dataloaders['train'].sampler)  # 当前轮的总正确率
            time_elapsed = time.time() - since

            print()
            print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format('train', epoch_loss, counter, epoch_acc))

            for phase in ['valid']:
                model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in tqdm(dataloaders['valid']):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with ema.average_parameters():
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum()
                    # 每个batch加1次
                    total_step[phase] += 1
                epoch_loss = running_loss / len(dataloaders['valid'].sampler)  # 当前轮的总体平均损失值
                epoch_acc = float(running_corrects) / len(dataloaders['valid'].sampler)  # 当前轮的总正确率
                time_elapsed = time.time() - since
                print()
                print('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format('valid', epoch_loss, counter, epoch_acc))



