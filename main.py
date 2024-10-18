
import tensorflow as tf
from demo import demo
from model import ModelTrain,ModelValid

flags = tf.app.flags
flags.DEFINE_string('MODE','demo', 'Set program to run in different mode, include train, valid and demo.')
flags.DEFINE_string('checkpoint_dir','./ckpt', 'Path to model file.')
flags.DEFINE_string('train_data','./data/fer2024/fer2024.csv','Path to training data.')
flags.DEFINE_string('valid_data','./valid_sets/','Path to training data.')
flags.DEFINE_boolean('show_box','False','Results will show detection box when true')

FLAGS = flags.FLAGS

def main(_):
    assert FLAGS.MODE in ['train','valid','demo']

    if FLAGS.MODE=='train':
        ModelTrain.train(FLAGS)

    elif FLAGS.MODE=='valid':
        ModelValid.valid(FLAGS)

    elif FLAGS.MODE=='demo':
        demo(FLAGS.chckpoint_dir,FLAGS.show_box)


if __name__ == '__main__':
    main()