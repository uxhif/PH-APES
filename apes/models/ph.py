import numpy as np
import gudhi as gd
import gudhi.representations
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import concurrent.futures
import os

# tf.config.set_visible_devices([], 'GPU') 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def ReduceDim(v, dim):
#     input_layer = Input(shape=(v.shape[1],))
#     encoded = Dense(dim, activation='relu')(input_layer)
#     decoded = Dense(v.shape[1], activation='relu')(encoded)

#     autoencoder = Model(input_layer, decoded)
#     encoder = Model(input_layer, encoded)

#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#     # reduced_v = encoder.predict(v, verbose=0).reshape(dim, 1)
#     reduced_v = encoder.predict(v, verbose=0).reshape(v.shape[0], dim, 1)

#     return reduced_v


def ComputePersistenceLandscape(points):
    # print(points.shape)
    # print(points.dtype, points.shape)
    ac = gd.AlphaComplex(points=points)
    stree = ac.create_simplex_tree()
    stree.compute_persistence() # need to compute dgm

    dgm_1 = stree.persistence_intervals_in_dimension(1) 
    dgm_2 = stree.persistence_intervals_in_dimension(2)

    LS = gd.representations.Landscape(resolution=1000)

    L_1 = LS.fit_transform([dgm_1])
    feature_1 = L_1[0].flatten().reshape(-1, 1)

    L_2 = LS.fit_transform([dgm_2])
    feature_2 = L_2[0].flatten().reshape(-1, 1)

    feature = np.hstack([feature_1, feature_2]).reshape(-1, 1)

    # print((feature.T).shape)
    return feature # (5000,1)

    # reduced_feature = ReduceDim(feature, 1024)
    # return reduced_feature # (1024, 1)


def ComputePersistenceSilhouette(points):
    ac = gd.AlphaComplex(points=points)
    stree = ac.create_simplex_tree()
    dgm = stree.persistence()

    SH = gd.representations.Silhouette(resolution=1000, weight=lambda x: np.power(x[1]-x[0],1))
    sh = SH.fit_transform([stree.persistence_intervals_in_dimension(1)])

    feature = sh[0].flatten().reshape(1, -1)

    reduced_feature = ReduceDim(feature, 1024)

    return reduced_feature


def ComputePersistenceImage(points):
    ac = gd.AlphaComplex(points=points)
    stree = ac.create_simplex_tree()
    dgm = stree.persistence()

    PI = gd.representations.PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1]**2, im_range=[0,.004,0,.004], resolution=[100,100])
    pi = PI.fit_transform([stree.persistence_intervals_in_dimension(1)])

    feature = pi[0].flatten().reshape(1, -1) #10000

    reduced_feature = ReduceDim(feature, 1024)

    return reduced_feature

method_dict = {
    'landscape': ComputePersistenceLandscape,
    'silhouette': ComputePersistenceSilhouette,
    'image': ComputePersistenceImage
}

def ComputeBatchFeatures(points, method='landscape'):
    global method_dict
    func = method_dict[method]
    return func(points)
    # return func(points.T)


def PersistenceHomology(x, method='landscape'):
    # global method_dict
    # B = x.shape[0]
    # PHfeature = []

    if method not in method_dict:
        raise ValueError(f"Method {method} is not recognized. Available methods: {list(method_dict.keys())}")

    # func = method_dict[method]

    # for i in range(B):
    #     # points = x[i]
    #     points = x[i].T
    #     feature = func(points)
    #     # print(feature.shape)
    #     PHfeature.append(np.squeeze(feature))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        PHfeature = list(executor.map(ComputeBatchFeatures, x))


    PHfeature = np.array(PHfeature)
    PHfeature = np.squeeze(PHfeature)
    print(PHfeature.shape)
    
    return PHfeature



def ExtractAndSave(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    for file in files:
        # Load point cloud data
        points = np.load(os.path.join(input_dir, file))
        if points.shape != (2048,3) or points.dtype != 'float32':
            print(file)
            assert False
        # Compute persistent homology feature
        feature = ComputePersistenceLandscape(points)
        # Save feature to a file
        output_file = os.path.join(output_dir, file)
        np.save(output_file, feature)

    # points = np.load(input_dir)
    # feature = ComputePersistenceLandscape(points)
    # # feature = PersistenceHomology(points.reshape(1,2048,3))
    # filename = os.path.basename(input_dir)
    # output_file = os.path.join(output_dir, filename.replace('.npy', '_ph.npy'))
    # np.save(output_file, feature)

# Set the directory paths
# input_directory = '/APES/data/shapenet/pcd/test'
# output_directory = '/APES/data/shapenet/ph_ls_2/test'
input_directory = '/APES/data/shapenet/pcd/train'
output_directory = '/APES/data/shapenet/ph_ls_2/train'

# tf.config.set_visible_devices([], 'GPU') 
# Extract and save feature
ExtractAndSave(input_directory, output_directory)
