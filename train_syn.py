"""
Usage:
  train.py [--gpu_id=<str>] [--dataset_filename=<str>] [--save_filename=<str>] \
  [--num_epochs=<N>] [--lr=<float>] [--learning_rate_decay=<float>] [--momentum_ratio=<float>]\
  [--cost_type=<str>] [--dropout_rate=<float>] [--lamb=<float>] [--neg_lamb=<float>] \
  [--epsilon=<float>][--xi=<float>][--eps_w=<float>][--n_eps_c=<float>][--n_eps_w=<float>] \
  [--norm_constraint=<str>][--num_power_iter=<N>] \
  [--monitoring_LDS] [--vis][--num_power_iter_for_monitoring_LDS=<N>]
  train.py -h | --help

Options:
  -h --help                                 Show this screen.
  --gpu_id=<str>                            [default: -1].
  --dataset_filename=<str>                  [default: syndata_1.pkl]
  --save_filename=<str>                     [default: trained_model.pkl]
  --num_epochs=<N>                          num_epochs [default: 1000].
  --lr=<float>                              lr [default: 1.0].
  --learning_rate_decay=<float>             learning_rate_decay [default: 0.995].
  --momentum_ratio=<float>                  [default: 0.9].
  --cost_type=<str>                         cost_type [default: MLE].
  --dropout_rate=<float>                    [default: 0.0].
  --lamb=<float>                            [default: 1.0].
  --neg_lamb=<float>                        [default: 1.0].
  --epsilon=<float>                         [default: 0.5].
  --xi=<float>                              [default: 0.000001].
  --eps_w=<float>                           [default: 0.5].
  --n_eps_c=<float>                         [default: 0.0].
  --n_eps_w=<float>                         [default: 0.0].
  --norm_constraint=<str>                   [default: L2].
  --num_power_iter=<N>                      [default: 1].
  --size=<N>                                [default: 1000].
  --monitoring_LDS
  --vis
  --num_power_iter_for_monitoring_LDS=<N>   [default: 5].
"""

import os
from docopt import docopt
arg = docopt(__doc__)
if int(arg["--gpu_id"]) != "-1":
    os.environ['THEANO_FLAGS'] = "device=cuda%s,floatX=float32" % arg["--gpu_id"]

import numpy
import theano
import theano.tensor as T
from six.moves import cPickle

from source import optimizers
from source import costs
from models.fnn_syn import FNN_syn
from models.fnn_syn_dropout import FNN_syn_dropout

import os
import sys
import errno
import traceback
from CostFunc import get_cost_type, get_cost_type_semi
from ExpUtils import ExpSaver, wlog


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def train(args):
    with open('dataset/' + args['--dataset_filename'], "rb") as fp:
        if sys.version_info.major == 3:
            dataset = cPickle.load(fp, encoding="bytes")
        else:
            dataset = cPickle.load(fp)

    x_train = theano.shared(numpy.asarray(dataset[0][0][0], dtype=theano.config.floatX))
    t_train = theano.shared(numpy.asarray(dataset[0][0][1], dtype='int32'))
    x_test = theano.shared(numpy.asarray(dataset[0][1][0], dtype=theano.config.floatX))
    t_test = theano.shared(numpy.asarray(dataset[0][1][1], dtype='int32'))

    avg_error_rate = 0
    train_err_history = 0
    test_err_history = 0
    exp = 1
    best_error_rate = 1000
    best_model = None
    statuses = {}

    for i in range(exp):
        numpy.random.seed(i*10)

        if args['--cost_type'] == 'dropout':
            model = FNN_syn_dropout(drate=float(args['--dropout_rate']))
        else:
            model = FNN_syn()
        x = T.matrix()
        t = T.ivector()
        ul_x = T.matrix()

        cost = get_cost_type_semi(model, x, t, ul_x, args)
        nll = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_test)
        error = costs.error(x=x, t=t, forward_func=model.forward_test)

        optimizer = optimizers.MomentumSGD(cost=cost, params=model.params, lr=float(args['--lr']),
                                           momentum_ratio=float(args['--momentum_ratio']))

        f_train = theano.function(inputs=[], outputs=cost, updates=optimizer.updates, givens={x: x_train, t: t_train, ul_x: x_test},
                                  on_unused_input='warn')
        f_nll_train = theano.function(inputs=[], outputs=nll, givens={x: x_train, t: t_train})
        f_nll_test = theano.function(inputs=[], outputs=nll, givens={x: x_test, t: t_test})
        f_error_train = theano.function(inputs=[], outputs=error, givens={x: x_train, t: t_train})
        f_error_test = theano.function(inputs=[], outputs=error, givens={x: x_test, t: t_test})
        if args['--monitoring_LDS']:
            LDS = costs.average_LDS_finite_diff(x,
                                                model.forward_test,
                                                main_obj_type='CE',
                                                epsilon=float(args['--epsilon']),
                                                norm_constraint=args['--norm_constraint'],
                                                num_power_iter=int(args['--num_power_iter_for_monitoring_LDS']))
            f_LDS_train = theano.function(inputs=[], outputs=LDS, givens={x: x_train})
            f_LDS_test = theano.function(inputs=[], outputs=LDS, givens={x: x_test})
        f_lr_decay = theano.function(inputs=[], outputs=optimizer.lr,
                                     updates={optimizer.lr: theano.shared(numpy.array(args['--learning_rate_decay']).astype(
                                         theano.config.floatX)) * optimizer.lr})

        statuses = {'nll_train': [], 'error_train': [], 'nll_test': [], 'error_test': []}
        if args['--monitoring_LDS']:
            statuses['LDS_train'] = []
            statuses['LDS_test'] = []

        statuses['nll_train'].append(f_nll_train())
        statuses['error_train'].append(f_error_train())
        statuses['nll_test'].append(f_nll_test())
        statuses['error_test'].append(f_error_test())
        if args['--monitoring_LDS']:
            statuses['LDS_train'].append(f_LDS_train())
            statuses['LDS_test'].append(f_LDS_test())
            # print("LDS_train : ", statuses['LDS_train'][-1], "LDS_test : ", statuses['LDS_test'][-1])
        for epoch in range(int(args['--num_epochs'])):
            f_train()
            if (epoch + 1) % 10 == 0:
                statuses['nll_train'].append(f_nll_train())
                statuses['error_train'].append(f_error_train())
                statuses['nll_test'].append(f_nll_test())
                statuses['error_test'].append(f_error_test())
                print("[Epoch]", str(epoch))
                print("nll_train : ", statuses['nll_train'][-1], "nll_test : ", statuses['nll_test'][-1], "error_test : ", statuses['error_test'][-1])
                if args['--monitoring_LDS']:
                    statuses['LDS_train'].append(f_LDS_train())
                    statuses['LDS_test'].append(f_LDS_test())
                    # print("LDS_train : ", statuses['LDS_train'][-1], "LDS_test : ", statuses['LDS_test'][-1])
            f_lr_decay()

        train_err_history += numpy.array(statuses['error_train']) * 1.0 / dataset[0][0][1].shape[0]
        test_err_history += numpy.array(statuses['error_test']) * 1.0 / dataset[0][1][1].shape[0]
        error_rate = statuses['error_test'][-1].item() * 1.0 / dataset[0][1][1].shape[0]
        # print("error rate", error_rate)
        if error_rate < best_error_rate:
            best_model = model
            best_error_rate = error_rate
        avg_error_rate += error_rate

    train_err_history /= exp
    test_err_history /= exp
    # saver.save_npy(train_err_history, "train_errrate")
    # saver.save_npy(test_err_history, "test_errrate")
    extra_info = "%s_%s_%s_%s" % (args['--epsilon'], args['--eps_w'], args['--n_eps_c'], args['--n_eps_w'])
    print("%s-avg error rate-%s-%g" % (extra_info, args['--save_filename'].split(".")[0], avg_error_rate / exp))
    print("%s-best error rate-%s-%g" % (extra_info, args['--save_filename'].split(".")[0], best_error_rate))
    make_sure_path_exists("./best_trained_model")
    cPickle.dump((best_model, statuses, args), open('./best_trained_model/' + args['--save_filename'], 'wb'), cPickle.HIGHEST_PROTOCOL)
    # saver.finish_exp()


if __name__ == '__main__':
    arg["exp"] = "avg"
    arg["dataset"] = arg["--dataset_filename"].split(".")[0]
    saver = ExpSaver("VATPlus-%s" % arg["--cost_type"], arg, ["--epsilon", "--eps_w", "--lamb"], None)
    if arg["--vis"]:
        saver.init_writer(["--epsilon", "--eps_w", "--lamb"])
    arg["log_dir"] = saver.log_dir
    run_time = saver.log_dir
    wlog("run time marker %s" % run_time)
    wlog("args in this experiment %s", '\n'.join(str(e) for e in sorted(vars(arg).items())))
    # noinspection PyBroadException
    try:
        train(arg)
    except BaseException as err:
        traceback.print_exc()
        saver.delete_dir()
        sys.exit(-1)
    saver.finish_exp({"num_epochs": 10})
