import numpy as np
import tensorflow as tf
import numbers

from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import constant_op


#----------------------------------------#
#-------------static methods-------------#
#----------------------------------------#


#-------------dropout_v3-------------#
# ~/miniconda3/envs/qsubnets/lib/python3.9/site-packages/tensorflow/python/ops/nn_ops.py
# and modified to get a fixed mask at prediction
def dropout_v3(x, rate, mask, name=None, default_name="drop"):
  """Shared implementation of the various dropout functions.

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    name: A name for this operation (optional).
    default_name: a default name in case `name` is `None`.

  Returns:
    A Tensor of the same shape and dtype of `x`.
  """
  with ops.name_scope(name, default_name, [x]) as name:
    is_rate_number = isinstance(rate, numbers.Real)
    if is_rate_number and (rate < 0 or rate >= 1):
      raise ValueError("`rate` must be a scalar tensor or a float in the "
                       f"range [0, 1). Received: rate={rate}")
    x = ops.convert_to_tensor(x, name="x")
    x_dtype = x.dtype
    if not x_dtype.is_floating:
      raise ValueError(
          "`x.dtype` must be a floating point tensor as `x` will be "
          f"scaled. Received: x_dtype={x_dtype}")

    is_executing_eagerly = context.executing_eagerly()
    if not tensor_util.is_tf_type(rate):
      if is_rate_number:
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
        ret = gen_math_ops.mul(x, scale)
      else:
        raise ValueError(
            f"`rate` must be a scalar or scalar tensor. Received: rate={rate}")
    else:
      rate.get_shape().assert_has_rank(0)
      rate_dtype = rate.dtype
      if rate_dtype != x_dtype:
        if not rate_dtype.is_compatible_with(x_dtype):
          raise ValueError(
              "`x.dtype` must be compatible with `rate.dtype`. "
              f"Received: x.dtype={x_dtype} and rate.dtype={rate_dtype}")
        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
      one_tensor = constant_op.constant(1, dtype=x_dtype)
      ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))
      
    # bob
    np_mask = np.empty((x.get_shape()[0], mask.size), dtype=bool)
    np_mask[:,:] = mask
    my_keep_mask = constant_op.constant(np_mask)
    zero_tensor = constant_op.constant(0, dtype=x_dtype)
    ret = array_ops.where_v2(my_keep_mask, ret, zero_tensor)
    
    if not is_executing_eagerly:
      ret.set_shape(x.get_shape())
    return ret



#-------------------------------------------------------#
#-------------class Dropout_Fixed_Pred_Mask-------------#
#-------------------------------------------------------#
# class Dropout_Fixed_Pred_Mask(tf.python.keras.layers.core.Dropout):
class Dropout_Fixed_Pred_Mask(tf.keras.layers.Dropout):
    """Class for Dropout Layer with a fixed mask at prediction.

    This layer is intended to be used in ANNs that are not trained.

    :param mode: when equals to `origin`, a new mask is sampled for each call to the layer (at construction). When equals to `pred` the associated mask is used (it is fixed)
    :type mode: str either `origin` or `pred`
    :param predict_mask: mask associated to the layer and used at prediction
    :type predict_mask: np.ndarray
    :param mask_init: when equals True `predict_mask` has already been initialized, when equals False `predict_mask` has not already been initialized yet
    :type mask_init: bool
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, rate, mode):
        super(Dropout_Fixed_Pred_Mask, self).__init__(rate)
        assert mode=="origin" or mode=="pred"

        self.__mode=mode
        self.__predict_mask=None
        self.__mask_init=False


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_mode-------------#
    def _get_mode(self):
        print("[Dropout_Fixed_Pred_Mask.py] Impossible to get the mode")
        return None

    #-------------_set_mode-------------#
    def _set_mode(self,new_mode):
        assert new_mode=="origin" or new_mode=="pred" or new_mode=="pred-std"
        self.__mode = new_mode

    #-------------_del_mode-------------#
    def _del_mode(self):
        print("[Dropout_Fixed_Pred_Mask.py] Impossible to delete the mode")

    #-------------_get_mask_init-------------#
    def _get_mask_init(self):
        print("[Dropout_Fixed_Pred_Mask.py] Impossible to get the mask initialization state")
        return None

    #-------------_set_mask_init-------------#
    def _set_mask_init(self,new_mask_init):
      print("[Dropout_Fixed_Pred_Mask.py] Impossible to set the mask initialization state")

    #-------------_del_mask_init-------------#
    def _del_mask_init(self):
        print("[Dropout_Fixed_Pred_Mask.py] Impossible to delete the mask initialization state")

    #-------------_get_predict_mask-------------#
    def _get_predict_mask(self):
        print("[Dropout_Fixed_Pred_Mask.py] Impossible to get the mask used for prediction")
        return None

    #-------------_set_predict_mask-------------#
    def _set_predict_mask(self,new_predict_mask):
      print("[Dropout_Fixed_Pred_Mask.py] Impossible to set the mask used for prediction")

    #-------------_del_predict_mask-------------#
    def _del_predict_mask(self):
        print("[Dropout_Fixed_Pred_Mask.py] Impossible to delete the mask used for prediction")
        
    #-------------property-------------#
    mode=property(_get_mode, _set_mode, _del_mode)
    mask_init=property(_get_mask_init, _set_mask_init, _del_mask_init)
    predict_mask=property(_get_predict_mask, _set_predict_mask, _del_predict_mask)
    

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------call-------------#
    # /home/gbriffoteaux/miniconda3/envs/qsubnets/lib/python3.9/site-packages/tensorflow/python/keras/layers/core.py
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=True):
      """Redefinition of the call() method from tf.keras.layers.Dropout"""
      
      # at construction, define the mask associated with this Dropout layer
      if not self.__mask_init:
        self.__mask_init=True
        self.__predict_mask = np.random.choice([True, False], size=(1, inputs.get_shape()[1]), p=[1.0-self.rate, self.rate])

      # at construction, usual Dropout
      if self.__mode=="origin":
        output = tf.keras.layers.Dropout.call(self, inputs, True)
      # at prediction, the mask is fixed (self.__predict_mask[0,:] is used)
      else:
        output = dropout_v3(x=inputs, rate=self.rate, mask=self.__predict_mask[0,:])

      return output
    
