from models/model import Model
import argparse
#from utils import *

#Parsing and configuration
def parse_args():
    desc = "Tensorflow implimentation of ___"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--mode', type=str, default='train', help='train or test mode?')
    parser.add_argument('--data', type=str, default='data', help='Which dataset to train on')

    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=250, help='The size of a batch')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Dir name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',help='Directory to save training logs')

    return check_args(parser.parse_args())

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
