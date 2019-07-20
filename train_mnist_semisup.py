"""
Usage:
  train.py [--gpu_id=<str>] [--save_filename=<str>] \
  [--num_epochs=<N>] [--batch_size==<N>] [--ul_batch_size=<N>] [--num_batch_it=<N>] \
  [--initial_learning_rate=<float>] [--learning_rate_decay=<float>] \
  [--layer_sizes=<str>] \
  [--cost_type=<str>] \
  [--xi=<float>]\
  [--dropout_rate=<float>] [--lamb=<float>] [--epsilon=<float>] [--norm_constraint=<str>] [--num_power_iter=<N>] \
  [--num_labeled_samples=<N>] [--num_validation_samples=<N>] \
  [--seed=<N>] [--vis] [--top_bn]
  train.py -h | --help

Options:
  -h --help                                 Show this screen.
  --gpu_id=<str>                            [default: -1].
  --save_filename=<str>                     [default: trained_model]
  --num_epochs=<N>                          num_epochs [default: 100].
  --batch_size=<N>                          batch_size [default: 100].
  --ul_batch_size=<N>                       ul_batch_size [default: 250].
  --num_batch_it=<N>                        num_batch_iteration [default: 500].
  --initial_learning_rate=<float>           initial_learning_rate [default: 0.002].
  --learning_rate_decay=<float>             learning_rate_decay [default: 0.9].
  --layer_sizes=<str>                       layer_sizes [default: 784-1200-1200-10]
  --cost_type=<str>                         cost_type [default: MLE].
  --lamb=<float>                            [default: 1.0].
  --epsilon=<float>                         [default: 2.0].
  --xi=<float>                              [default: 0.000001].
  --norm_constraint=<str>                   [default: L2].
  --num_power_iter=<N>                      [default: 1].
  --num_labeled_samples=<N>                 [default: 100].
  --num_validation_samples=<N>              [default: 1000].
  --seed=<N>                                [default: 1].
  --vis
  --top_bn
"""

import traceback

import numpy
from docopt import docopt

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from dllib.ExpUtils import *

arg = docopt(__doc__)

if arg["--gpu_id"] == "":
    arg["--gpu_id"] = auto_select_gpu()

if int(arg["--gpu_id"]) != "-1":
    os.environ['THEANO_FLAGS'] = "device=cuda%s,floatX=float32" % arg["--gpu_id"]


import theano
import theano.tensor as t_func
from theano.tensor.shared_randomstreams import RandomStreams

from dllib.ExpUtils import *
from CostFunc import get_cost_type_semi
from source import optimizers
from source import costs
from models.fnn_mnist_semisup import FNN_MNIST
from collections import OrderedDict
import load_data

from tensorboardX import SummaryWriter

run_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
dir_path = os.path.join(os.environ['HOME'], 'project/runs', 'VAT-theano-semi/mnist/%s_xi=1e-6_top_bn=%s_%s' % (arg['--cost_type'],
                                                                                                               str(arg['--top_bn']), run_time))
# writer = SummaryWriter(log_dir=dir_path)


def train(args):
    print(args)
    numpy.random.seed(int(args['--seed']))

    dataset = load_data.load_mnist_for_semi_sup(n_l=int(args['--num_labeled_samples']),
                                                n_v=int(args['--num_validation_samples']))

    x_train, t_train, ul_x_train = dataset[0]
    x_test, t_test = dataset[2]

    layer_sizes = [int(layer_size) for layer_size in args['--layer_sizes'].split('-')]
    model = FNN_MNIST(layer_sizes=layer_sizes)

    x = t_func.matrix()
    ul_x = t_func.matrix()
    t = t_func.ivector()

    cost_semi = get_cost_type_semi(model, x, t, ul_x, args)
    nll = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_test)
    error = costs.error(x=x, t=t, forward_func=model.forward_test)

    optimizer = optimizers.ADAM(cost=cost_semi, params=model.params, alpha=float(args['--initial_learning_rate']))

    index = t_func.iscalar()
    ul_index = t_func.iscalar()
    batch_size = int(args['--batch_size'])
    ul_batch_size = int(args['--ul_batch_size'])

    f_train = theano.function(inputs=[index, ul_index], outputs=cost_semi, updates=optimizer.updates,
                              givens={
                                  x: x_train[batch_size * index:batch_size * (index + 1)],
                                  t: t_train[batch_size * index:batch_size * (index + 1)],
                                  ul_x: ul_x_train[ul_batch_size * ul_index:ul_batch_size * (ul_index + 1)]},
                              on_unused_input='ignore')
    f_nll_train = theano.function(inputs=[index], outputs=nll,
                                  givens={
                                      x: x_train[batch_size * index:batch_size * (index + 1)],
                                      t: t_train[batch_size * index:batch_size * (index + 1)]})
    f_nll_test = theano.function(inputs=[index], outputs=nll,
                                 givens={
                                     x: x_test[batch_size * index:batch_size * (index + 1)],
                                     t: t_test[batch_size * index:batch_size * (index + 1)]})

    f_error_train = theano.function(inputs=[index], outputs=error,
                                    givens={
                                        x: x_train[batch_size * index:batch_size * (index + 1)],
                                        t: t_train[batch_size * index:batch_size * (index + 1)]})
    f_error_test = theano.function(inputs=[index], outputs=error,
                                   givens={
                                       x: x_test[batch_size * index:batch_size * (index + 1)],
                                       t: t_test[batch_size * index:batch_size * (index + 1)]})

    f_lr_decay = theano.function(inputs=[], outputs=optimizer.alpha,
                                 updates={optimizer.alpha: theano.shared(
                                     numpy.array(args['--learning_rate_decay']).astype(
                                         theano.config.floatX)) * optimizer.alpha})

    # Shuffle training set
    randix = RandomStreams(seed=numpy.random.randint(1234)).permutation(n=x_train.shape[0])
    update_permutation = OrderedDict()
    update_permutation[x_train] = x_train[randix]
    update_permutation[t_train] = t_train[randix]
    f_permute_train_set = theano.function(inputs=[], outputs=x_train, updates=update_permutation)

    # Shuffle unlabeled training set
    ul_randix = RandomStreams(seed=numpy.random.randint(1234)).permutation(n=ul_x_train.shape[0])
    update_ul_permutation = OrderedDict()
    update_ul_permutation[ul_x_train] = ul_x_train[ul_randix]
    f_permute_ul_train_set = theano.function(inputs=[], outputs=ul_x_train, updates=update_ul_permutation)

    statuses = {'nll_train': [], 'error_train': [], 'nll_test': [], 'error_test': []}

    n_train = x_train.get_value().shape[0]
    n_test = x_test.get_value().shape[0]
    n_ul_train = ul_x_train.get_value().shape[0]

    l_i = 0
    ul_i = 0
    for epoch in range(int(args['--num_epochs'])):
        # cPickle.dump((statuses, args), open('./trained_model/' + 'tmp-' + args['--save_filename'], 'wb'),
        #              cPickle.HIGHEST_PROTOCOL)
        f_permute_train_set()
        f_permute_ul_train_set()
        for it in range(int(args['--num_batch_it'])):
            f_train(l_i, ul_i)
            l_i = 0 if l_i >= n_train / batch_size - 1 else l_i + 1
            ul_i = 0 if ul_i >= n_ul_train / ul_batch_size - 1 else ul_i + 1

        sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in range(int(n_train / batch_size))])) * batch_size
        sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in range(int(n_train / batch_size))]))
        sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in range(int(n_test / batch_size))])) * batch_size
        sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in range(int(n_test / batch_size))]))
        statuses['nll_train'].append(sum_nll_train / n_train)
        statuses['error_train'].append(sum_error_train)
        statuses['nll_test'].append(sum_nll_test / n_test)
        statuses['error_test'].append(sum_error_test)
        wlog("[Epoch] %d" % epoch)
        acc = 1 - 1.0*statuses['error_test'][-1]/n_test
        wlog("nll_test : %f error_test : %d accuracy:%f" % (statuses['nll_test'][-1], statuses['error_test'][-1], acc))
        # writer.add_scalar("Test/Loss", statuses['nll_test'][-1], epoch * int(args['--num_batch_it']))
        # writer.add_scalar("Test/Acc", acc, epoch * int(args['--num_batch_it']))
        f_lr_decay()
    # fine_tune batch stat
    f_fine_tune = theano.function(inputs=[ul_index], outputs=model.forward_for_finetuning_batch_stat(x),
                                  givens={x: ul_x_train[ul_batch_size * ul_index:ul_batch_size * (ul_index + 1)]})
    [f_fine_tune(i) for i in range(n_ul_train // ul_batch_size)]

    sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in range(n_test // batch_size)])) * batch_size
    sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in range(n_test // batch_size)]))
    statuses['nll_test'].append(sum_nll_test / n_test)
    statuses['error_test'].append(sum_error_test)
    acc = 1 - 1.0*statuses['error_test'][-1]/n_test
    wlog("final nll_test: %f error_test: %d accuracy:%f" % (statuses['nll_test'][-1], statuses['error_test'][-1], acc))
    # writer.add_scalar("Test/Loss", statuses['nll_test'][-1], epoch * int(args['--num_batch_it']))
    # writer.add_scalar("Test/Acc", acc, epoch * int(args['--num_batch_it']))


if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        train(arg)
    except BaseException as err:
        traceback.print_exc()
        sys.exit(-1)
