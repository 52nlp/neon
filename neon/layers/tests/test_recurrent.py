# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from nose.plugins.attrib import attr

from neon.layers import RecurrentHiddenLayer
from neon.layers import RecurrentLSTMLayer
from neon.backends import gen_backend # Alex said not to use this. How should CPU/GPU be imported/tested?
from neon.params import UniformValGen
#GPU = gen_backend(rng_seed=0, gpu='nervanagpu')


nin = 3
nout = 2
unrolls = 5
batch_size = 10
weight_init_rec = UniformValGen # this should be the default for recurrent layers


def test_fprop(layer, inputs):
    for step in range(unrolls):
        y = layer.output_list[step-1]
        layer.fprop(y, None, inputs, step)


def test_bprop(layer, errors):
    for step in range(unrolls):
        y = layer.output_list[step-1]
        layer.bprop(y, None, errors, step)


class TestRecurrentHiddenLayer():

    def initialize_layer(self, backend):
        weight_init_rec = UniformValGen(backend=backend)
        layer = RecurrentHiddenLayer(nin=nin,
                                    nout=nout,
                                    unrolls=unrolls,
                                    batch_size=batch_size,
                                    backend=backend,
                                    weight_init_rec=weight_init_rec)
        layer.initialize([])
        return layer


    def test_cpu_fprop(self):
        backend = gen_backend(rng_seed=0)
        layer = self.initialize_layer(backend)
        inputs = backend.ones((nin, batch_size))
        test_fprop(layer, inputs)

    def test_cpu_bprop(self):
        backend = gen_backend(rng_seed=0)
        layer = self.initialize_layer(backend)
        errors = backend.ones((nin, batch_size))
        test_bprop(layer, errors)

    @attr('cuda')
    def test_gpu_fprop(self):
        pass

    @attr('cuda')
    def test_gpu_bprop(self):
        pass


class TestRecurrentLSTMLayer():

    def initialize_layer(self, backend):
        weight_init_rec = UniformValGen(backend=backend)
        layer = RecurrentLSTMLayer(nin=nin,
                                   nout=nout,
                                   unrolls=unrolls,
                                   batch_size=batch_size,
                                   backend=backend,
                                   gate_activation=Tanh() # this should be a default
                                   weight_init_rec=weight_init_rec)
        layer.initialize([])
        return layer


    def test_cpu_fprop(self):
        backend = gen_backend(rng_seed=0)
        layer = self.initialize_layer(backend)
        inputs = backend.ones((nin, batch_size))
        test_fprop(layer, inputs)

    def test_cpu_bprop(self):
        backend = gen_backend(rng_seed=0)
        layer = self.initialize_layer(backend)
        errors = backend.ones((nin, batch_size))
        test_bprop(layer, errors)

    @attr('cuda')
    def test_gpu_fprop(self):
        pass

    @attr('cuda')
    def test_gpu_bprop(self):
        pass
