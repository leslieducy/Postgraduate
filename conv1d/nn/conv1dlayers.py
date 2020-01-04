import numpy as np


def conv_forward_bak(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    N, _, height, width = padding_z.shape
    print(padding_z.shape)
    C, D, k1, k2 = K.shape
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    print(conv_z.shape)
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z


if __name__ == "__main__":
    z = np.array(range(2*3*5*5))
    z = z.reshape(2,3,5,5)
    K = np.array(range(3*4*3*3))
    K = K.reshape(3,4,3,3)
    b = np.array([1,1,1,1])
    conv_z = conv_forward_bak(z, K, b, padding=(1, 1), strides=(1, 1))
    print(conv_z)