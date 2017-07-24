from __future__ import division
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
  num_train = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]
  scores = np.zeros([num_train,num_classes])
  exp_scores = np.zeros([num_train,num_classes])
  probs = np.zeros([num_train,num_classes])
  dscores = np.zeros([num_train,num_classes])
  correct_logprobs = np.zeros([num_train,1])
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
      for j in range(num_classes):
          scores[i,j] = np.dot(X[i,:],W[:,j])
          exp_scores[i,j] = np.exp(scores[i,j])
      probs[i,:] = exp_scores[i,:] / float(np.sum(exp_scores[i,:]))
      correct_logprobs[i] = -np.log(probs[i,y[i]])
      dscores[i,:] = probs[i,:]
      dscores[i,y[i]] -= 1
  dscores /= num_train

  for i in range(num_features):
      dW[i,:] = np.dot(X[:,i].T,dscores)
      
  loss += np.sum(correct_logprobs)/float(num_train)
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True)
  correct_logprobs = -np.log(probs[range(num_train),y])
  data_loss = np.sum(correct_logprobs)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  
  loss += data_loss + reg_loss
  
  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  dW = np.dot(X.T,dscores)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

