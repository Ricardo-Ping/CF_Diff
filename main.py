import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import models.gaussian_diffusion as gd

from models.CAM_AE import CAM_AE
from models.CAM_AE_multihops import CAM_AE_multihops

import evaluate_utils
import data_utils
import random

random_seed = 1
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ML-1M', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[256,256]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

print("torch version:", torch.__version__)

args = parser.parse_args()
print("args:", args)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

### DATA LOAD ###
data_name = 'ML-1M'
train_path = args.data_path + 'train_list_' + args.dataset + '.npy'
valid_path = args.data_path + 'valid_list_' + args.dataset + '.npy'
test_path = args.data_path + 'test_list_' + args.dataset + '.npy'

n_hop = 3  # The number of hops neighbors, e.g. n_hop=3 means three hops neighbors are taken into account
print("{}-hop neighbors are taken into account".format(n_hop))
if n_hop == 2:
    sec_hop = torch.load(args.data_path + 'sec_hop_inters_ML_1M.pt')
    multi_hop = sec_hop
elif n_hop == 3:
    multi_hop = torch.load(args.data_path + 'multi_hop_inters_ML_1M.pt')

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0,
                          worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

train_loader_sec_hop = DataLoader(multi_hop, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0,
                                  worker_init_fn=worker_init_fn)
test_loader_sec_hop = DataLoader(multi_hop, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data

print('data is ready.')

### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
                                 args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

# Build model
if n_hop == 2:
    model = CAM_AE(16, 2, 2, n_item, args.emb_size).to(device)
elif n_hop == 3:
    model = CAM_AE_multihops(16, 4, 2, n_item, args.emb_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models are ready.")


def evaluate(data_loader, data_loader_sec_hop, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(data_loader), enumerate(data_loader_sec_hop)):
            his_data = mask_his[e_idxlist[batch_idx * args.batch_size:batch_idx * args.batch_size + len(batch)]]
            batch = batch.to(device)
            batch_2 = batch_2.to(device)
            prediction = diffusion.p_sample(model, batch, batch_2, args.sampling_steps, args.sampling_noise)
            prediction[his_data.nonzero()] = -np.inf
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results


if __name__ == '__main__':

    best_recall, best_epoch = -100, 0
    best_test_result = None
    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= 20:
            print('-' * 18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0

        for (batch_idx, batch), (batch_idx_2, batch_2) in zip(enumerate(train_loader), enumerate(train_loader_sec_hop)):
            batch = batch.to(device)
            batch_2 = batch_2.to(device)
            batch_count += 1
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, batch_2, args.reweight)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            valid_results = evaluate(test_loader, test_loader_sec_hop, valid_y_data, train_data, eval(args.topN))
            if args.tst_w_val:
                test_results = evaluate(test_twv_loader, test_loader_sec_hop, test_y_data, mask_tv, eval(args.topN))
            else:
                test_results = evaluate(test_loader, test_loader_sec_hop, test_y_data, mask_tv, eval(args.topN))
            evaluate_utils.print_results(None, valid_results, test_results)

            if valid_results[1][1] > best_recall:  # recall@20 as selection
                best_recall, best_epoch = valid_results[1][1], epoch
                best_results = valid_results
                best_test_results = test_results

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model,
                           '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                           .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims,
                                   args.emb_size, args.mean_type, \
                                   args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps,
                                   args.reweight, args.log_name))

        print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
            "%H: %M: %S", time.gmtime(time.time() - start_time)))
        print('---' * 18)

    print('===' * 18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    evaluate_utils.print_results(None, best_results, best_test_results)
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
