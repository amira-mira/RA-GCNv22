import numpy as np


class Data_transform():
    def __init__(self, data_transform):
        self.data_transform = data_transform

    def __call__(self, data):
        x, location = data
        if self.data_transform:
            C, T, V, M = x.shape
            x_new = np.zeros((C*3, T, V, M))
            x_new[:C,:,:,:] = x
            for i in range(T-1):
                x_new[C:(2*C),i,:,:] = x[:,i+1,:,:] - x[:,i,:,:]
            for i in range(V):
                x_new[(2*C):,:,i,:] = x[:,:,i,:] - x[:,:,1,:]
            return (x_new, location)
        else:
            return (x, location)


class Occlusion_part():
    def __init__(self, occlusion_part):
        self.occlusion_part = occlusion_part

        self.parts = dict()
        self.parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1              # left arm
        self.parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1           # right arm
        self.parts[3] = np.array([22, 23, 24, 25]) - 1                  # two hands
        self.parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1  # two legs
        self.parts[5] = np.array([1, 2, 3, 4, 21]) - 1                  # trunk

    def __call__(self, data):
        x, location = data
        for part in self.occlusion_part:
            x[:,:,self.parts[part],:] = 0
        return (x, location)


class Occlusion_time():
    def __init__(self, occlusion_time):
        self.occlusion_time = int(occlusion_time // 2)

    def __call__(self, data):
        x, location = data
        if not self.occlusion_time == 0:
            x[:,(50-self.occlusion_time):(50+self.occlusion_time),:,:] = 0
        return (x, location)


class Occlusion_block():
    def __init__(self, threshold):
        if threshold == 0:
            self.threshold = 0
        else:
            self.threshold = 50 * (threshold + 2)

    def __call__(self, data):
        x, location = data
        if self.threshold:
            y_max = np.max(location[1,:,:,:])
            mask = location[1] > (y_max - self.threshold)
            for i in range(x.shape[0]):
                x[i][mask] = 0
        return (x, location)


class Occlusion_rand():
    def __init__(self, occlusion_rand, data_shape):
        C, T, V, M = data_shape
        self.mask = np.random.rand(T, V, M)
        self.mask[self.mask > occlusion_rand] = 1
        self.mask[self.mask <= occlusion_rand] = 0

    def __call__(self, data):
        x, location = data
        x = x * self.mask[np.newaxis, :, :, :]
        return (x, location)


class Jittering_joint():
    def __init__(self, jittering_joint, data_shape, sigma=1):
        C, T, V, M = data_shape
        noise = sigma * np.random.randn(T, V, M)
        self.mask = np.random.rand(T, V, M)
        self.mask[self.mask > jittering_joint] = 1
        self.mask[self.mask <= jittering_joint] = 0
        self.mask = 1 - self.mask
        self.mask *= noise

    def __call__(self, data):
        x, location = data
        x = x + self.mask[np.newaxis, :, :, :]
        return (x, location)


class Jittering_frame():
    def __init__(self, jittering_frame, data_shape, sigma=1):
        C, T, V, M = data_shape
        noise = sigma * np.random.randn(T)
        self.mask = np.random.rand(T)
        self.mask[self.mask > jittering_frame] = 1
        self.mask[self.mask <= jittering_frame] = 0
        self.mask = 1 - self.mask
        self.mask *= noise

    def __call__(self, data):
        x, location = data
        x = x + self.mask[np.newaxis, :, np.newaxis, np.newaxis]
        return (x, location)

