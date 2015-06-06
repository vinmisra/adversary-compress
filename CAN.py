"""
Code for compressive adversarial network implementation. Based on "Generative Adversarial Networks", by Goodfellow et al
"""
import sgd
import functools
wraps = functools.wraps
import itertools
import numpy
np = numpy
import theano
import warnings
theano.config.compute_test_value = 'off'

from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.space import VectorSpace, IndexSpace
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
from pylearn2.train import SerializationGuard

import train_AE

class CompressAdversaryPair(Model):

    def __init__(self, compressor, discriminator, 
                monitor_compressor=False, 
                monitor_discriminator=False):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self

    def get_params(self):
        p = self.compressor.get_params() + self.discriminator.get_params()
        return p

    def get_input_space(self):
        return self.compressor.get_input_space()

    def get_target_space(self):
        return IndexSpace(max_labels=11, dim=1)

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        X,Y = data
        Xhat = self.compressor.reconstruct(X)

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

        space = CompositeSpace([self.get_input_space(),self.get_target_space()])
        source = (self.get_input_source(), self.get_target_source())
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

    def get_monitoring_data_specs(self):
        return self.mlp.get_monitoring_data_specs()

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

class AdversaryCost_A(Cost):

    # Combination of both internally generated labels (true or fake --- veracity) and ground truth labels (number betwen 0 and 9 if true --- precision)
    # 11 total categories: fake, and 0-9.
    # generator seeks to max p(d(fake)=label), while discriminator seeks to max p(d(.)=.)
    def get_data_specs(self,model):
        space = CompositeSpace([model.get_input_space(),model.get_target_space()])
        sources = ('features','targets')
        return (space,sources)


    def __init__(self, 
            init_train_clock=1.,
            discriminator_steps=1,
            joint_steps=0,
            compressor_steps=0,
            ever_train_compressor=0,
            ever_train_discriminator=1
            ):
        self.__dict__.update(locals())
        del self.self
        # These allow you to dynamically switch off training parts.
        # If the corresponding ever_train_* is False, these have
        # no effect.
        self.train_clock = sharedX(numpy.array(init_train_clock,dtype='float32')) #increments throughout training cycle

        if init_train_clock > discriminator_steps:
            self.now_train_compressor = sharedX(numpy.array(1.,dtype='float32'))
        else:
            self.now_train_compressor = sharedX(numpy.array(0.,dtype='float32'))

        if init_train_clock > discriminator_steps+joint_steps:
            self.now_train_discriminator = sharedX(numpy.array(0., dtype='float32'))
        else:
            self.now_train_discriminator = sharedX(numpy.array(1.,dtype='float32'))


    def expr(self, model, data, **kwargs):
        d_obj, g_obj = self.get_objectives(model, data)
        l = []
        # This stops stuff from ever getting computed if we're not training
        # it.
        if self.ever_train_discriminator:
            l.append(d_obj)
        if self.ever_train_compressor:
            l.append(g_obj)
        return sum(l)

    def get_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, CompressAdversaryPair)
        g = model.compressor
        d = model.discriminator

        # Data should be of the form (data,labels)
        X,Y = data
        
        #testing
        # X.tag.test_value = numpy.random.random([5,784]).astype(numpy.float32)
        # ytest = numpy.random.randint(low=10,size=[5,1]).astype(numpy.float32)
        # Y.tag.test_value = ytest


        #generate inputs
        X_pure = X
        X_reconstructed = g.reconstruct(X)
        # if self.noise_both != 0.:
        #     rng = MRG_RandomStreams(2014 / 6 + 2)
        #     S = S + rng.normal(size=S.shape, dtype=S.dtype) * self.noise_both
        #     X = X + rng.normal(size=X.shape, dtype=S.dtype) * self.noise_both

        # create our semi-artificial labels
        Y_pure = Y#T.concatenate([Y,T.alloc(0.,Y.shape[0],1)],axis=1)
        Y_reconstructed = T.alloc(10,Y.shape[space.get_batch_axis()],1)

        #generate predictions 
        yhat_pure = d.fprop(X_pure)
        yhat_reconstructed = d.fprop(X_reconstructed)

        #softmax is likely to be the last layer for categorical Y, so below calls softmax.cost = log loss
        d_obj = 0.5 * (d.layers[-1].cost(Y_pure, yhat_pure) + d.layers[-1].cost(Y_reconstructed, yhat_reconstructed)) 
        g_obj = d.layers[-1].cost(Y_pure, yhat_reconstructed)


        # if self.no_drop_in_d_for_g:
        #     y_hat0_no_drop = d.dropout_fprop(S)
        #     g_obj = d.layers[-1].cost(y1, y_hat0_no_drop)
        # else:
        #     g_obj = d.layers[-1].cost(y1, y_hat0)

        # if self.blend_obj:
        #     g_obj = (self.zurich_coeff * g_obj - self.minimax_coeff * d_obj) / (self.zurich_coeff + self.minimax_coeff)

        return d_obj, g_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, CompressAdversaryPair)
        g = model.compressor
        d = model.discriminator

        #get raw gradients for d and g objectives...
        d_obj, g_obj = self.get_objectives(model, data)
        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        # if self.scale_grads:
        #     S_grad = T.grad(g_obj, S)
        #     scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
        #     g_grads = [g_grad * scale for g_grad in g_grads]

        #adjust raw gradients with control signals
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

        #update control signals using the updates return functionality
        updates = OrderedDict()
        #first, the clock
        self.future_train_clock = T.switch(T.ge(self.train_clock,self.discriminator_steps+self.joint_steps+self.compressor_steps),1.,self.train_clock+1.)
        updates[self.train_clock] = self.future_train_clock
        #then the control signals
        updates[self.now_train_discriminator] = T.switch(T.le(self.future_train_clock,self.discriminator_steps+self.joint_steps),1.,0.)
        updates[self.now_train_compressor] = T.switch(T.gt(self.future_train_clock,self.discriminator_steps),1.,0.)

        return rval, updates

    def get_monitoring_channels(self, model, data, **kwargs):

        X_pure,Y_pure = data
        X_pure.tag.test_value = numpy.random.random(size=[5,784]).astype('float32')
        Y_pure.tag.test_value = numpy.random.randint(10,size=[5,1]).astype('int64')
        rval = OrderedDict()

        g = model.compressor
        d = model.discriminator

        yhat_pure = T.argmax(d.fprop(X_pure),axis=1).dimshuffle(0,'x')
        yhat_reconstructed = T.argmax(d.fprop(g.reconstruct(X_pure)),axis=1).dimshuffle(0,'x')

        rval['conviction_pure'] = T.cast(T.eq(yhat_pure,10).mean(), 'float32')
        rval['accuracy_pure'] = T.cast(T.eq(yhat_pure,Y_pure).mean(), 'float32')
        rval['inaccuracy_pure'] = 1 - rval['conviction_pure']-rval['accuracy_pure']

        rval['conviction_fake'] = T.cast(T.eq(yhat_reconstructed,10).mean(), 'float32')
        rval['accuracy_fake'] = T.cast(T.eq(yhat_reconstructed,Y_pure).mean(), 'float32')
        rval['inaccuracy_fake'] = 1 - rval['conviction_fake']-rval['accuracy_fake']

        rval['discernment_pure'] = rval['accuracy_pure']+rval['inaccuracy_pure']
        rval['discernment_fake'] = rval['conviction_fake']
        rval['discernment'] = 0.5*(rval['discernment_pure']+rval['discernment_fake'])

        # y = T.alloc(0., m, 1)  
        d_obj, g_obj = self.get_objectives(model, data)
        rval['objective_d'] = d_obj
        rval['objective_g'] = g_obj

        #monitor probability of true
        # rval['now_train_compressor'] = self.now_train_compressor
        return rval       

class save_pieces(TrainExtension):
    def __init__(self, save_path, *args, **kwargs):
        super(save_pieces,self).__init__(*args,**kwargs)
        self.save_path = save_path

    def on_monitor(self, model,dataset,algorithm):
        try:
            # Make sure that saving does not serialize the dataset
            dataset._serialization_guard = SerializationGuard()
            print "Saving model files to " + self.save_path
            #save whole model
            serial.save(self.save_path, model,
                        on_overwrite='backup')
            #save compressor and discriminator
            serial.save(self.save_path+".cmp", model.compressor)
            serial.save(self.save_path+".dis", model.discriminator)
            print "Finished saving."
        finally:
            dataset._serialization_guard = None
