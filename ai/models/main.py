from model import Model
import tensorflow as tf
import argparse
#from utils import *

#Parsing and configuration
def parse_args():
    desc = "Tensorflow implimentation of ___"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--mode', type=str, default='train', help='train or test mode?')
    parser.add_argument('--restore', type=bool, default=False, help='To restore model')

    parser.add_argument('--img_size', type=int, default=10, help='The dim of the images')
    parser.add_argument('--z_dim', type=int, default=3, help='Z dim of the images')
    parser.add_argument('--label_dim', type=int, default=10, help='Num output categories')
    parser.add_argument('--channels', type=int, default=64, help='Number of conv channels')

    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=10, help='The size of a batch')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--use_bias', type=bool, default=True, help='Bias term')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Dir name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',help='Directory to save training logs')
    return parser.parse_args()
  #  return check_args(parser.parse_args())
'''
#Checking args
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('Epochs must be >= 1')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be >= 1')
    return args
'''
#Main
def main():
    #parse arguments
    args = parse_args()
    if args is None:
        exit()

    #Run session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        model = Model(sess, args)

        #Build the comp Graph
        model.build_model()

        #Need to add method for showing variables
        if args.mode == 'train':

            #Launch graph in a seession
            model.train()
            print '[*] training has finished'

            model.test()
            print '[*] testing has fninished'

        if args.mode == 'test':
            model.test()
            print '[*] Test has finished'

if __name__ == '__main__':
    main()
