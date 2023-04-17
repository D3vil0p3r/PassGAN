import argparse
import base64, zlib
import sample, train

def print_banner():
    #cat banner.txt | gzip | base64
    encoded_data = "H4sIAAAAAAAAA2VOQQ7AMAi6+wqOW7LEDzVxD/HxA9puh5GIliAWAIrAD7Xhh4UgJzuG9fqQmAy0LM3K0JQUvXLbskiN3DaiJQR2hlYOnW3R+UaUjc5HpI/pI1el90xjjgp1a9viAQ+uT3TmAAAA"
    banner = zlib.decompress(base64.b64decode(encoded_data), 16 + zlib.MAX_WBITS).decode('utf-8')
    print(banner)

def help():
   # Display Help
   print_banner()
   print("A Deep Learning Approach for Password Guessing.\n")

   print("List of arguments:\n")
   
   print("-h, --help              show this help message and exit")
   print("sample                  use the pretrained model to generate passwords")
   print("train                   train a model on a large dataset (can take several hours on a GTX 1080)")
   print("")
   print("Usage Examples:")
   print("passgan sample --input-dir pretrained --checkpoint pretrained/checkpoints/195000.ckpt --output gen_passwords.txt --batch-size 1024 --num-samples 1000000")
   print("passgan train --output-dir output --training-data data/train.txt")

def arg_parse():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action='store_true', help="show this help message and exit")
    subparsers = parser.add_subparsers(dest="cmd")
    sample_subparser = subparsers.add_parser('sample', help='use the pretrained model to generate passwords')
    train_subparser = subparsers.add_parser('train', help='train a model on a large dataset (can take several hours on a GTX 1080)')

    sample_subparser.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    sample_subparser.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .ckpt file.')

    sample_subparser.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    sample_subparser.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    sample_subparser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    sample_subparser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    sample_subparser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')

    train_subparser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    train_subparser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    train_subparser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    train_subparser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    train_subparser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    train_subparser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    train_subparser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    train_subparser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    train_subparser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    
    print(args)
    
    if args.help or (not(args.cmd)):
        help()
        exit()
    
    if args.cmd == "sample":
        print(vars(args))
        #sample
    
    if args.cmd == "train":
        print(vars(args))
        #train