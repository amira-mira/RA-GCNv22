import numpy as np
from matplotlib import pyplot as plt


class Visualizer():
    def __init__(self, args):
        self.visualize_sample = args.visualize_sample
        self.visualize_class = args.visualize_class
        self.stream = args.visualize_stream
        self.frames = args.visualize_frames

        data = np.load('./visualize.npz')
        weight = data['weight']
        feature = data['feature']
        out = data['out']
        self.location = data['location']
        self.label = data['label']
        print('\n*********************Video Name************************')
        print(data['name'][self.visualize_sample])

        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.visualize_sample] + 1
        self.actural_class = self.label[self.visualize_sample] + 1
        if self.visualize_class == 0:
            self.visualize_class = self.actural_class
        self.probably_value = out[self.visualize_sample, self.visualize_class-1]

        self.location = self.location[self.visualize_sample,:,:,:,:]
        weight = weight[self.stream]
        feature = feature[self.stream]
        feature = feature[self.visualize_sample,:,:,:,:]
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)   # CAM method
        self.result = self.result[self.visualize_class-1,:,:25,:]


    def show_skeleton(self):
        C, T, V, M = self.location.shape
        connecting_joint = np.array([2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12])

        result = np.maximum(self.result, 0)
        result = result/np.max(result)

        if len(self.frames) > 0:
            self.location = self.location[:,self.frames,:,:]
            plt.figure()

            for t in range(len(self.frames)):
                plt.cla()
                plt.xlim(-50, 2000)
                plt.ylim(-50, 1100)
                plt.title('visualize_sample:{}, visualize_class:{}, stream:{}, frame:{}\nprobably_value:{:.4f}%, pred_class:{}, actural_class:{}'.format(
                          self.visualize_sample, self.visualize_class, self.stream, self.frames[t], self.probably_value*100, self.pred_class, self.actural_class))

                for m in range(M):
                    x = self.location[0,t,:,m]
                    y = 1080 - self.location[1,t,:,m]

                    col = []
                    for v in range(V):
                        r = r = result[int(self.frames[t]/4),v,m]
                        g = 0
                        b = 1 - r
                        col.append([r, g, b])
                        k = connecting_joint[v] - 1
                        plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=(0.1,0.1,0.1), linewidth=0.5, markersize=0)
                    plt.scatter(x, y, marker='o', c=col, s=16)
                plt.show()

        else:
            plt.figure()
            plt.ion()

            for t in range(self.location.shape[1]):
                if np.sum(self.location[:,t,:,:]) == 0:
                    break

                plt.cla()
                plt.xlim(-50, 2000)
                plt.ylim(-50, 1100)
                plt.title('visualize_sample:{}, visualize_class:{}, stream:{}, frame:{}\nprobably_value:{:.4f}%, pred_class:{}, actural_class:{}'.format(
                          self.visualize_sample, self.visualize_class, self.stream, t, self.probably_value*100, self.pred_class, self.actural_class))

                for m in range(M):
                    x = self.location[0,t,:,m]
                    y = 1080 - self.location[1,t,:,m]

                    col = []
                    for v in range(V):
                        r = result[int(t/4),v,m]
                        g = 0
                        b = 1 - r
                        col.append([r, g, b])
                        k = connecting_joint[v] - 1
                        plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=(0.1,0.1,0.1), linewidth=0.5, markersize=0)
                    plt.scatter(x, y, marker='o', c=col, s=16)
                plt.pause(0.1)
            plt.ioff()
            plt.show()


    def show_heatmap(self):
        plt.figure()
        plt.suptitle('visualize_sample: {}, visualize_class: {}, stream: {}\nprobably_value: {:.4f}%, pred_class: {}, actural_class: {}'.format(
                     self.visualize_sample, self.visualize_class, self.stream, self.probably_value*100, self.pred_class, self.actural_class))

        plt.subplot(1,2,1)
        plt.ylabel('Frames')
        plt.xlabel('Skeletons: first person')
        plt.imshow(self.result[:,:,0], cmap=plt.cm.plasma, vmin=0, vmax=np.max(self.result))
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.ylabel('Frames')
        plt.xlabel('Skeletons: second person')
        plt.imshow(self.result[:,:,1], cmap=plt.cm.plasma, vmin=0, vmax=np.max(self.result))
        plt.colorbar()

        plt.show()


    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        print('\n*********************Wrong Sample**********************')
        print(wrong_sample)


    def show_important_joints(self):
        first_sum = np.sum(self.result[:,:,0], axis=0)
        second_sum = np.sum(self.result[:,:,1], axis=0)
        first_index = np.argsort(-first_sum) + 1
        second_index = np.argsort(-second_sum) + 1

        print('\n*********************First Person**********************')
        print('Weights of all joints:')
        print(first_sum)
        print('\nMost important joints:')
        print(first_index)
        print('\n*********************Second Person*********************')
        print('Weights of all joints:')
        print(second_sum)
        print('\nMost important joints:')
        print(second_index)
        print('\n')

