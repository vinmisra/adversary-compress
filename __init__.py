"""
Code for compressive adversarial network implementation. Based on "Generative Adversarial Networks", by Goodfellow et alÏ
"""
import functools
wraps = functools.wraps
import itertools
import numpy
np = numpy
import theano
import warnings

from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.space import VectorSpace
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
from pylearn2.models.mlp import Layer
from pylearn2.models.mlp import Linear
from pylearn2.models import Model
from pylearn2.space import CompositeSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import block_gradient
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX

class CompressAdversaryPair(Model):

    def __init__(self, compressor, discriminator):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self

    def get_params(self):
        p = self.compressor.get_params() + self.discriminator.get_params()
        return p

    def get_input_space(self):
        return self.compressor.get_input_space()

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        X,Y = data
        Xhat = self.compressor.fprop(X)

        c_ch = self.compressor.get_monitoring_channels(X)
        d_ch = self.discriminator.get_monitoring_channels((X,Y))
        d_distorted_ch = self.discriminator.get_monitoring_channels((Xhat, Y))

        if self.monitor_compressor:
            for key in c_ch:
                rval['compress_' + key] = c_ch[key]
        if self.monitor_discriminator:
            for key in d_ch:
                rval['dis_on_data_' + key] = d_ch[key]
            for key in d_ch:
                rval['dis_on_distorted_' + key] = d_distorted_ch[key]
        return rval

    def get_monitoring_data_specs(self):

        space = self.discriminator.get_input_space()
        source = self.discriminator.get_input_source()
        return (space, source)

    def _modify_updates(self, updates):
        self.compressor.modify_updates(updates)
        self.discriminator.modify_updates(updates)

    def get_lr_scalers(self):
        rval = self.compressor.get_lr_scalers()
        rval.update(self.discriminator.get_lr_scalers())
        return rval

class compressor(Model):
    #very simple wrapper around MLP

    def __init__(self, mlp):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2015 * 4 * 20)

    def get_input_space(self):
        return self.mlp.get_input_space()

    def reconstruct(self, input):
        return self.mlp.fprop(input)

    def get_monitoring_channels(self, data):
        rval = OrderedDict()
        try:
            rval.update(self.mlp.get_monitoring_channels(data))
        except Exception:
            warnings.warn("something went wrong with compressor.mlp's monitoring channels")
        return rval

    def get_params(self):
        return self.mlp.get_params()

    def get_output_space(self):
        return self.mlp.get_output_space()

    def _modify_updates(self, updates):
        self.mlp.modify_updates(updates)

    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()

    def __setstate__(self, state):
        self.__dict__.update(state)

class AdversaryCost_VeracityPrecision(DefaultDataSpecsMixin, Cost):

    # Combination of both internally generated labels (true or fake --- veracity) and ground truth labels (number betwen 0 and 9 if true --- precision)
    # 11 total categories: fake, and 0-9.
    # generator seeks to max p(d(fake)=label), while discriminator seeks to max p(d(.)=.)
    supervised = True

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            compressor_default_input_include_prob = 1.,
            compressor_default_input_scale=1.,
            inference_default_input_include_prob=None,
            inference_input_include_probs=None,
            inference_default_input_scale=1.,
            inference_input_scales=None,
            init_now_train_compressor=True,
            ever_train_discriminator=True,
            ever_train_compressor=True,
            ever_train_inference=True,
            no_drop_in_d_for_g=False,
            alternate_g = False,
            infer_layer=None,
            noise_both = 0.,
            blend_obj = False,
            minimax_coeff = 1.,
            zurich_coeff = 1.):
        self.__dict__.update(locals())
        del self.self
        # These allow you to dynamically switch off training parts.
        # If the corresponding ever_train_* is False, these have
        # no effect.
        self.now_train_compressor = sharedX(init_now_train_compressor)
        self.now_train_discriminator = sharedX(numpy.array(1., dtype='float32'))
        self.now_train_inference = sharedX(numpy.array(1., dtype='float32'))

    def expr(self, model, data, **kwargs):
        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        l = []
        # This stops stuff from ever getting computed if we're not training
        # it.
        if self.ever_train_discriminator:
            l.append(d_obj)
        if self.ever_train_compressor:
            l.append(g_obj)
        if self.ever_train_inference:
            l.append(i_obj)
        return sum(l)

    def get_samples_and_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.compressor
        d = model.discriminator

        # Note: this assumes data is design matrix
        X = data
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        # NOTE: if this changes to optionally use dropout, change the inference
        # code below to use a non-dropped-out version.
        S, z, other_layers = g.sample_and_noise(m, default_input_include_prob=self.compressor_default_input_include_prob, default_input_scale=self.compressor_default_input_scale, all_g_layers=(self.infer_layer is not None))

        if self.noise_both != 0.:
            rng = MRG_RandomStreams(2014 / 6 + 2)
            S = S + rng.normal(size=S.shape, dtype=S.dtype) * self.noise_both
            X = X + rng.normal(size=X.shape, dtype=S.dtype) * self.noise_both

        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))

        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = d.dropout_fprop(S)
            g_obj = d.layers[-1].cost(y1, y_hat0_no_drop)
        else:
            g_obj = d.layers[-1].cost(y1, y_hat0)

        if self.blend_obj:
            g_obj = (self.zurich_coeff * g_obj - self.minimax_coeff * d_obj) / (self.zurich_coeff + self.minimax_coeff)

        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            pred = model.inferer.dropout_fprop(S_nograd, self.inference_default_input_include_prob,
                                                self.inference_input_include_probs,
                                                self.inference_default_input_scale,
                                                self.inference_input_scales)
            if self.infer_layer is None:
                target = z
            else:
                target = other_layers[self.infer_layer]
            i_obj = model.inferer.layers[-1].cost(target, pred)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.compressor
        d = model.discriminator

        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict()
        zeros = itertools.repeat(theano.tensor.constant(0., dtype='float32'))
        if self.ever_train_discriminator:
            rval.update(OrderedDict(safe_zip(d_params, [self.now_train_discriminator * dg for dg in d_grads])))
        else:
            rval.update(OrderedDict(zip(d_params, zeros)))
        if self.ever_train_compressor:
            rval.update(OrderedDict(safe_zip(g_params, [self.now_train_compressor * gg for gg in g_grads])))
        else:
            rval.update(OrderedDict(zip(g_params, zeros)))
        if self.ever_train_inference and model.inferer is not None:
            i_params = model.inferer.get_params()
            i_grads = T.grad(i_obj, i_params)
            rval.update(OrderedDict(safe_zip(i_params, [self.now_train_inference * ig for ig in i_grads])))
        elif model.inferer is not None:
            rval.update(OrderedDict(model.inferer.get_params(), zeros))

        updates = OrderedDict()

        # Two d steps for every g step
        if self.alternate_g:
            updates[self.now_train_compressor] = 1. - self.now_train_compressor

        return rval, updates

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.compressor
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())
        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if model.monitor_inference and i_obj != 0:
            rval['objective_i'] = i_obj
        if model.monitor_discriminator:
            rval['objective_d'] = d_obj
        if model.monitor_compressor:
            rval['objective_g'] = g_obj

        rval['now_train_compressor'] = self.now_train_compressor
        return rval
