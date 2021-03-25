import os
import shutil
import pynvml
import torch


def check_gpu(gpus):
    if len(gpus) > 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        for i in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memused = meminfo.used / 1024 / 1024
            print('GPU{} used: {}M'.format(i, memused))
            if memused > 1000:
                pynvml.nvmlShutdown()
                raise ValueError('GPU{} is occupied!'.format(i))
        pynvml.nvmlShutdown()
        return torch.device('cuda')
    else:
        print('Using CPU!')
        return torch.device('cpu')


def load_checkpoint(fname='checkpoint'):
    fpath = '/home/aayadi/RA-GCNv22/models/' + fname + '.pth.tar'
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath)
        return checkpoint
    else:
        raise ValueError('Do NOT exist this checkpoint: {}'.format(fname))


def save_checkpoint(model, optimizer, epoch, best, is_best, model_name):
    if not os.path.exists('/home/aayadi/RA-GCNv22/models'):
        os.mkdir('/home/aayadi/RA-GCNv22/models')
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {'model':model, 'optimizer':optimizer, 'epoch':epoch, 'best':best}
    torch.save(checkpoint, '/home/aayadi/RA-GCNv22/models/checkpoint.pth.tar')
    if is_best:
        shutil.copy('/home/aayadi/RA-GCNv22/models/checkpoint.pth.tar', '/home/aayadi/RA-GCNv22/models/' + model_name + '.pth.tar')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot recognize the input parameter {}'.format(v))

