#!/usr/bin/env python3

import os
import time
import pickle
import argparse
import base64, zlib

#####

import sys
sys.path.append(os.getcwd())

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import ctypes
import ctypes.util
name = ctypes.util.find_library('cudart64_110.dll')
lib = ctypes.cdll.LoadLibrary(name)

# name = ctypes.util.find_library('cublas64_11.dll')
# lib = ctypes.cdll.LoadLibrary(name)
 
# name = ctypes.util.find_library('cublasLt64_11.dll')
# lib = ctypes.cdll.LoadLibrary(name)

name = ctypes.util.find_library('cufft64_10.dll')
lib = ctypes.cdll.LoadLibrary(name)

name = ctypes.util.find_library('curand64_10.dll')
lib = ctypes.cdll.LoadLibrary(name)

# name = ctypes.util.find_library('cusolver64_11.dll')
# lib = ctypes.cdll.LoadLibrary(name)

name = ctypes.util.find_library('cusparse64_11.dll')
lib = ctypes.cdll.LoadLibrary(name)

# name = ctypes.util.find_library('cudnn64_8.dll')
# lib = ctypes.cdll.LoadLibrary(name)

def print_banner():
    #cat banner.txt | gzip | base64
    encoded_data = "H4sIAAAAAAAAA3VRSQ6AMAi884o5amLCh0zqQ3i8zGBdqpJIcRbACvRoFcCIvEDrJEYxhkhtB7/o2wi9lOOgPGus4q5FmqMyEJREPm7lzSoJ2TbJjsQjc0iMIGCa2HvRNhEIpvls0yTWHGgr12AutTSXV2mtks11hGTGj/m73Pft2Q/1/AkqdlE7dKq0AQAA"
    banner = zlib.decompress(base64.b64decode(encoded_data), 16 + zlib.MAX_WBITS).decode('utf-8')
    print(banner)

print_banner()

print("\nLoading TensorFlow...\n")

import utils
import models
import tensorflow as tf
import numpy as np
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
print("")

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import tflib.plot


######### SAMPLE GENERATION #########

def sample_run(args):

    with open(os.path.join(args.input_dir, 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f)

    with open(os.path.join(args.input_dir, 'inv_charmap.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f)

    # print('LENGTH',args.batch_size, args.seq_length, args.layer_dim, len(charmap))
    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))

    with tf.compat.v1.Session() as session:

        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        def save(samples):
            with open(args.output, 'a') as f:
                    for s in samples:
                        s = "".join(s).replace('`', '')
                        f.write(s + "\n")

        saver = tf.compat.v1.train.Saver()
        saver.restore(session, args.checkpoint)

        samples = []
        then = time.time()
        start = time.time()
        for i in range(int(args.num_samples / args.batch_size)):

            samples.extend(generate_samples())

            # append to output file every 1000 batches
            if i % 1000 == 0 and i > 0: 

                save(samples)
                samples = [] # flush

                print('wrote {} samples to {} in {:.2f} seconds. {} total.'.format(1000 * args.batch_size, args.output, time.time() - then, i * args.batch_size))
                then = time.time()

        save(samples)
        print('finished in {:.2f} seconds'.format(time.time() - start))

###################################

######### TRAINING MODELS #########

def train_run(args):

    lines, charmap, inv_charmap = utils.load_dataset(
        path=args.training_data,
        max_length=args.seq_length,
    )

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'))

    if not os.path.isdir(os.path.join(args.output_dir, 'samples')):
        os.makedirs(os.path.join(args.output_dir, 'samples'))

    # pickle to avoid encoding errors with json
    with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
        pickle.dump(charmap, f)

    with open(os.path.join(args.output_dir, 'inv_charmap.pickle'), 'wb') as f:
        pickle.dump(inv_charmap, f)

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
    fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

    disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
    disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[args.batch_size,1,1],
        minval=0.,
        maxval=1.
    )

    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(models.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += args.lamb * gradient_penalty

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    # Dataset iterator
    def inf_train_gen():
        while True:
            np.random.shuffle(lines)
            for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
                yield np.array(
                    [[charmap[c] for c in l] for l in lines[i:i+args.batch_size]],
                    dtype='int32'
                )

    # During training we monitor JS divergence between the true & generated ngram
    # distributions for n=1,2,3,4. To get an idea of the optimal values, we
    # evaluate these statistics on a held-out set first.
    true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[10*args.batch_size:], tokenize=False) for i in range(4)]
    validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*args.batch_size], tokenize=False) for i in range(4)]
    for i in range(4):
        print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
    true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        gen = inf_train_gen()

        for iteration in range(args.iters):
            start_time = time.time()

            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op)

            # Train critic
            for i in range(args.critic_iters):
                _data = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete:_data}
                )

            lib.plot.output_dir = args.output_dir
            lib.plot.plot('time', time.time() - start_time)
            lib.plot.plot('train disc cost', _disc_cost)

            if iteration % 100 == 0 and iteration > 0:
                samples = []
                for i in range(10):
                    samples.extend(generate_samples())

                for i in range(4):
                    lm = utils.NgramLanguageModel(i+1, samples, tokenize=False)
                    lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

                with open(os.path.join(args.output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w') as f:
                    for s in samples:
                        s = "".join(s)
                        f.write(s + "\n")

            if iteration % args.save_every == 0 and iteration > 0:
                model_saver = tf.train.Saver()
                model_saver.save(session, os.path.join(args.output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))

            lib.plot.tick()

###################################


def main(args=None):

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    subp_sample = subparsers.add_parser("sample", add_help=False)

    subp_sample.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    subp_sample.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .ckpt file.')

    subp_sample.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    subp_sample.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    subp_sample.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    subp_sample.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    subp_sample.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')

    subp_train = subparsers.add_parser("train", add_help=False)

    subp_train.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    subp_train.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    subp_train.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    subp_train.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    subp_train.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    subp_train.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    subp_train.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    subp_train.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    subp_train.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')

    parsed_args = parser.parse_args(args)

    if parsed_args.cmd == "sample":
        
        if not os.path.isdir(parsed_args.input_dir):
            parser.error('"{}" folder doesn\'t exist'.format(parsed_args.input_dir))

        if not os.path.exists(parsed_args.checkpoint + '.meta'):
            parser.error('"{}.meta" file doesn\'t exist'.format(parsed_args.checkpoint))

        if not os.path.exists(os.path.join(parsed_args.input_dir, 'charmap.pickle')):
            parser.error('charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(parsed_args.input_dir))

        if not os.path.exists(os.path.join(parsed_args.input_dir, 'inv_charmap.pickle')):
            parser.error('inv_charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(parsed_args.input_dir))

        sample_run(parsed_args)
    elif parsed_args.cmd == "train":
        train_run(parsed_args)
    else:
        print("usage:")
        print("passgan.py sample --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --output gen_passwords.txt --batch-size 1024 --num-samples 1000000")
        print("passgan.py train --output-dir pretrained --training-data data/train.txt")

if __name__ == "__main__":
    main()