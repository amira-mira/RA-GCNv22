import os
import yaml
import argparse

import src.utils as U
from src.processor import Processor
from src.visualizer import Visualizer


def main():
    parser = Init_parameters()

    # Update parameters by yaml
    args = parser.parse_args()
    if os.path.exists('./configs/' + args.config + '.yaml'):
        with open('./configs/' + args.config + '.yaml', 'r') as f:
            yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist the parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this config: {}'.format(args.config))

    # Update parameters by cmd
    args = parser.parse_args()

    # Show parameters
    print('\n************************************************')
    print('The running config is presented as follows:')
    v = vars(args)
    for i in v.keys():
        print('{}: {}'.format(i, v[i]))
    print('************************************************\n')

    # Processing
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, args.gpus)))
    if args.visualization:
        if args.extract:
            p = Processor(args)
            p.extract()

        print('Starting visualizing ...')
        v = Visualizer(args)
        v.show_wrong_sample()
        v.show_important_joints()
        v.show_heatmap()
        v.show_skeleton()
        print('Finish visualizing!')
        
    else:
        p = Processor(args)
        p.start()


def Init_parameters():
    parser = argparse.ArgumentParser(description='Richly Activated Graph Convolutional Network (RA-GCN) for Skeleton-based Action Recognition')

    # Config
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)

    # Processing
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualization', '-v', default=False, action='store_true', help='Visualization')

    # Program
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')

    # Visualization
    parser.add_argument('--visualize_sample', '-vs', type=int, default=0, help='select sample from 0 ~ batch_size-1')
    parser.add_argument('--visualize_class', '-vc', type=int, default=0, help='select class from 0 ~ 60, 0 means actural class')
    parser.add_argument('--visualize_stream', '-vb', type=int, default=0, help='select stream from 0 ~ model_stream-1')
    parser.add_argument('--visualize_frames', '-vf', type=int, default=[], nargs='+', 
                        help='show specific frames from 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--subset', '-ss', type=str, default='cs', choices=['cs', 'cv'], help='benchmark of NTU dataset')
    parser.add_argument('--max_frame', '-mf', type=int, default=300, help='max frame number')
    parser.add_argument('--batch_size', '-bs', type=int, default=16, help='batch size')
    parser.add_argument('--data_transform', '-dt', type=U.str2bool, default=True, 
                        help='channel 0~2: original data, channel 3~5: next_frame - now_frame, channel 6~8: skeletons_all - skeleton_2')
    parser.add_argument('--occlusion_part', '-op', type=int, nargs='+', default=[], choices=[1, 2, 3, 4, 5], 
                        help='1:left arm, 2:right arm, 3:two hands, 4:two legs, 5:trunk')
    parser.add_argument('--occlusion_time', '-ot', type=int, default=0, 
                        help='0 to 100, number of occlusion frames in first 100 frames')
    parser.add_argument('--occlusion_block', '-ob', type=int, default=0, help='1 to 6, occlusion threshold')
    parser.add_argument('--occlusion_rand', '-or', type=float, default=0, help='probability of random occlusion')
    parser.add_argument('--jittering_joint', '-jj', type=float, default=0, help='probability of joint jittering')
    parser.add_argument('--jittering_frame', '-jf', type=float, default=0, help='probability of frame jittering')
    parser.add_argument('--sigma', type=float, default=0, help='std of jittering')

    # Model
    parser.add_argument('--pretrained', '-pt', type=U.str2bool, default=True, help='load pretrained baseline for each stream')
    parser.add_argument('--model_stream', '-ms', type=int, default=3, help='number of model streames')
    parser.add_argument('--gcn_kernel_size', '-ks', type=int, nargs='+', default=[5,2], help='[temporal_window_size, spatial_max_distance]')
    parser.add_argument('--drop_prob', '-dp', type=int, default=0.5, help='dropout probability')

    # Optimizer
    parser.add_argument('--max_epoch', '-me', type=int, default=50, help='max training epoch')
    parser.add_argument('--learning_rate', '-lr', type=int, default=0.1, help='initial learning rate')
    parser.add_argument('--adjust_lr', '-al', type=int, nargs='+', default=[10,30], help='divide learning rate by 10')

    return parser


if __name__ == '__main__':
    main()

