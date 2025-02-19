import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  for i in xrange(X.shape[0]):
    f_i = X[i].dot(W)
    f_i -= np.max(f_i)
    sum_j = np.sum(np.exp(f_i))
    loss += np.log(sum_j) - f_i[y[i]]
    for k in range(W.shape[1]):
      p_k = np.exp(f_i[k]) / sum_j
      dW[:, k] += (p_k - (k == y[i])) * X[i]
  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += np.sum((reg * W) * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims=True)
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
  p = np.exp(f) / sum_f
  loss = np.sum(-np.log(p[np.arange(X.shape[0]), y]))
  indicators = np.zeros_like(p)
  indicators[np.arange(X.shape[0]), y] = 1
  dW = X.T.dot(p - indicators)
  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += np.sum((reg * W) * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

