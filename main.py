import vgg, vgg_preprocessing
import tensorflow as tf
from datetime import datetime

batch_size = 25
query_file_num = 250
test_file_num = 1000
height, width = 224, 224
query_dir = "UKBench_small/query/"
test_dir = "UKBench_small/test/"
ckpt_dir = "checkpoint/vgg_16.ckpt"

query_output_dir = "outputs/query/"
test_output_dir = "outputs/test/"

def _make_filenames():
    filenames = []
    for i in range(test_file_num):
        filenames.append(test_dir+"image_"+"{0:05d}".format(i)+".jpg")
    for i in range(query_file_num):
        filenames.append(query_dir+"image_"+"{0:05d}".format(4*i)+".jpg")
    return filenames

# Data preperation phase
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    rgb = tf.cond(tf.equal(tf.shape(image_decoded)[2], 1),
                  lambda: tf.tile(image_decoded, [1, 1, 3]),
                  lambda: tf.identity(image_decoded))
    image_processed = vgg_preprocessing.preprocess_image(rgb, height, width)
    return image_processed

def _make_dataset(filenames):
    eval_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    eval_dataset = eval_dataset.map(lambda fname: (_parse_function(fname)),
                                    num_parallel_calls=batch_size)
    eval_dataset = eval_dataset.batch(batch_size)
    eval_dataset = eval_dataset.prefetch(1)
    eval_dataset = eval_dataset.make_one_shot_iterator()
    return eval_dataset

def _fc7_flat(eval_dataset):
    inputs = eval_dataset.get_next("inputs")
    logits, end_points = vgg.vgg_16(inputs,
                                    num_classes=1000,
                                    is_training=False,  # this time, we don't train the network.
                                    dropout_keep_prob=0.5,  # doesn't matter since "is_training=False."
                                    spatial_squeeze=True,  # squeeze spatial dimensions in the final output
                                    scope='vgg_16',
                                    fc_conv_padding='VALID',  # this code uses conv instead of fc (they are equivalent!)
                                    global_pool=False
                                    )
    fc7_features = end_points['vgg_16/fc7']
    return tf.layers.flatten(fc7_features, name="fc7_flat")

def _make_features():
    filenames = _make_filenames()
    eval_dataset = _make_dataset(filenames)
    return _fc7_flat(eval_dataset)

def _eval_features(sess, features):
    for i in range(test_file_num//batch_size):
        outputs = sess.run(features)
        np.save(test_output_dir+str(i), outputs)
        print("%s: test %d generated" %(str(datetime.now()), i*batch_size))
    print("%s: test features generated" %(str(datetime.now())))
    for i in range(query_file_num//batch_size):
        outputs = sess.run(features)
        np.save(query_output_dir+str(i), outputs)
        print("%s: query %d generated" %(str(datetime.now()), i*batch_size))
    print("%s: query features generated" %(str(datetime.now())))

fc7_features = _make_features()

if __name__ == "__main__":
    import numpy as np
    import time
    EVALUATION_FLAG = False
    CONTINUE_FLAG = False
    CONTINUE_FROM = 0
    CALCULATION_FLAG = True
    tf.logging.set_verbosity(30)

    saver = tf.train.Saver()
    net_sess = tf.Session()
    saver.restore(net_sess, ckpt_dir)

    distances = np.zeros((query_file_num, test_file_num))
    
    if CONTINUE_FLAG:
        distances = np.loadtxt("distances.txt")
    start_time = time.time()

    # feature evaluation stage(preprocess)
    if EVALUATION_FLAG:
        _eval_features(net_sess, fc7_features)
    net_sess.close()
    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - start_time 
    print("feature evaluation ended in %f seconds" %(preprocess_time))

    # distance calculation stage(run)
    run_sess = tf.Session()
    if CALCULATION_FLAG:
        for query_batch_index in range(CONTINUE_FLAG*CONTINUE_FROM//batch_size,query_file_num//batch_size):
            query_outputs_split = tf.split(np.load(query_output_dir+str(query_batch_index)+".npy"), batch_size)
            distances_temp = [[0 for j in range(test_file_num)] for i in range(batch_size)]
            for test_batch_index in range(test_file_num//batch_size):
                test_outputs_split = tf.split(np.load(test_output_dir+str(test_batch_index)+".npy"), batch_size)
                for query_index in range(batch_size):
                    for test_index in range(batch_size):
                        distances_temp[query_index][test_batch_index*batch_size+test_index] = tf.norm(query_outputs_split[query_index]-test_outputs_split[test_index], ord='euclidean')
                    # distances[query_index, test_index:test_index+batch_size] = sess.run(list(map(lambda x:tf.norm(query_output-x), test_outputs_split)))
                    # print("%s: Query %d upto %d finished" %(str(datetime.now()), query_batch_index*batch_size+query_index, (test_batch_index+1)*batch_size))
            print("%s: Query %d finished" %(str(datetime.now()), (query_batch_index+1)*batch_size))
            distances[query_batch_index*batch_size:(query_batch_index+1)*batch_size, :] = run_sess.run(distances_temp)
            tf.keras.backend.clear_session()
            np.savetxt("distances.txt", distances, fmt="%f")
    run_sess.close()
    calculation_end_time = time.time()

    # print time elapsed
    run_time = calculation_end_time - preprocess_end_time
    total_time = calculation_end_time - start_time
    print("proprocess: %f seconds" %(preprocess_time))
    print("run       : %f seconds" %(run_time))
    print("total     : %f seconds" %(total_time))

    # print accuracy
    distances_sorted = np.argsort(distances, axis=1)
    corrects = 0
    all_corrects = 0
    for i in range(distances.shape[0]):
        top4 = distances_sorted[i][0:4]
        top4_matches = map(lambda x: x>=0 and x<4, top4-4*i)
        corrects = corrects + sum(top4_matches)
        all_corrects = all_corrects + all(top4_match==True for top4_match in top4_matches)
    correct_ratio = corrects/distances.shape[0]/4
    all_correct_ratio = all_corrects/distances.shape[0]
    print("average accuracy    : %.4f" %correct_ratio)
    print("exact match accuracy: %.4f" %all_correct_ratio)
    
