import sys
from source import costs


def get_cost_type_semi(model, x, t, ul_x, args, fix_d=None):
    if args['--cost_type'] == 'MLE':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train)
    elif args['--cost_type'] == 'L2':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params, coeff=float(args['--lamb']))
    elif args['--cost_type'] == 'AT':
        cost = costs.adversarial_training(x, t, model.forward_train,
                                          'CE',
                                          epsilon=float(args['--epsilon']),
                                          lamb=float(args['--lamb']),
                                          norm_constraint=args['--norm_constraint'],
                                          forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif args['--cost_type'] == 'VATsup':
        cost = costs.virtual_adversarial_training_finite_diff(x, t, model.forward_train,
                                                              'CE',
                                                              epsilon=float(args['--epsilon']),
                                                              norm_constraint=args['--norm_constraint'],
                                                              num_power_iter=int(args['--num_power_iter']),
                                                              x_for_generating_adversarial_examples=x,
                                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif args['--cost_type'] == 'VAT':
        cost = costs.virtual_adversarial_training(x, t, model.forward_train,
                                                  'CE',
                                                  epsilon=float(args['--epsilon']),
                                                  norm_constraint=args['--norm_constraint'],
                                                  num_power_iter=int(args['--num_power_iter']),
                                                  x_for_generating_adversarial_examples=ul_x,
                                                  forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif args['--cost_type'] == 'VAT_f':
        cost = costs.virtual_adversarial_training_finite_diff(x, t, model.forward_train,
                                                              'CE',
                                                              xi=float(args['--xi']),
                                                              epsilon=float(args['--epsilon']),
                                                              norm_constraint=args['--norm_constraint'],
                                                              num_power_iter=int(args['--num_power_iter']),
                                                              x_for_generating_adversarial_examples=ul_x,
                                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    else:
        print("method not exists")
        sys.exit(-1)
    return cost


def get_cost_type(model, x, t, args):
    forward_func = None
    if args["dataset"] == "mnist":
        forward_func = model.forward_no_update_batch_stat
    if args['--cost_type'] == 'MLE' or args['--cost_type'] == 'dropout':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train)
    elif args['--cost_type'] == 'L2':
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params, coeff=float(args['--lamb']))
    elif args['--cost_type'] == 'AT':
        cost = costs.adversarial_training(x, t, model.forward_train,
                                          'CE',
                                          epsilon=float(args['--epsilon']),
                                          lamb=float(args['--lamb']),
                                          norm_constraint=args['--norm_constraint'],
                                          forward_func_for_generating_adversarial_examples=forward_func)
    elif args['--cost_type'] == 'VAT':
        cost = costs.virtual_adversarial_training(x, t, model.forward_train,
                                                  'CE',
                                                  epsilon=float(args['--epsilon']),
                                                  norm_constraint=args['--norm_constraint'],
                                                  num_power_iter=int(args['--num_power_iter']),
                                                  forward_func_for_generating_adversarial_examples=forward_func)
    elif args['--cost_type'] == 'VAT_n':
        cost = costs.virtual_adversarial_training_noise(x, t, model.forward_train,
                                                        'CE',
                                                        epsilon=float(args['--epsilon']),
                                                        eps_n=float(args['--eps_w']),
                                                        neg_lamb=float(args['--neg_lamb']),
                                                        norm_constraint=args['--norm_constraint'],
                                                        num_power_iter=int(args['--num_power_iter']),
                                                        forward_func_for_generating_adversarial_examples=forward_func)
    elif args['--cost_type'] == 'VAT_f':
        cost = costs.virtual_adversarial_training_finite_diff(x, t, model.forward_train,
                                                              'CE',
                                                              epsilon=float(args['--epsilon']),
                                                              norm_constraint=args['--norm_constraint'],
                                                              num_power_iter=int(args['--num_power_iter']),
                                                              forward_func_for_generating_adversarial_examples=forward_func)
    else:
        print("method not exists")
        sys.exit(-1)
    return cost
