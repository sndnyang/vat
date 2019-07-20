import theano.tensor as T


def error(x, t, forward_func):
    y = forward_func(x)
    return T.sum(T.neq(T.argmax(y, axis=1), t))
