#!/usr/bin/env python
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

import numpy as np
from nose.plugins.attrib import attr
from nose.tools import nottest

from neon.util.testing import assert_tensor_equal, assert_tensor_near_equal

def m_assert_tensor_equal(t1, t2):
    for _t1, _t2, ctx in zip(t1._tensorlist, t2._tensorlist, t1._ctxs):
        ctx.push()
        assert_tensor_equal(_t1, _t2)
        ctx.pop()

@attr('cuda')
class TestGPU(object):

    def setup(self):
        from neon.backends.mgpu import MGPU, MGPUTensor
        # this code gets called prior to each test
        self.be = MGPU(rng_seed=0, num_dev=2)
        self.gpt = MGPUTensor

    @attr('bbx')
    def reduction_test(self):
        nr = self.be.num_dev
        # create a numpy array as the test-bed
        ASIZE = 9
        # round up to the nearest multiple of num_dev
        BSIZE = -(-ASIZE // nr) * nr
        h_a = np.random.randn(ASIZE * nr).reshape(
                (nr, ASIZE)).astype(self.be.default_dtype)
        h_result = np.sum(h_a, axis=0, keepdims=True)

        d_a = self.be.empty((1, ASIZE))
        u_a = self.be.empty((1, BSIZE))
        self.be.scatter(h_a, d_a)
        self.be.reduce(d_a, u_a, async=True)
        print(h_result)
        print(d_a.tlis[0].asnumpyarray())

        for i in range(nr):
            np.testing.assert_allclose(d_a.tlis[i].asnumpyarray(),
                                       h_result, atol=1e-6, rtol=0)

    @attr('memset')
    def memset_test(self):
        nr = self.be.num_dev
        # create a numpy array as the test-bed
        ASIZE = 9
        # round up to the nearest multiple of num_dev
        BSIZE = -(-ASIZE // nr) * nr

        d_a = self.be.zeros((1, ASIZE))
        print(d_a.tlist[0].asnumpyarray())

    @attr('frag2rep')
    def frag2rep_test(self):
        nr = self.be.num_dev
        np.random.seed(0)
        # create a numpy array as the test-bed
        (rows, cols) = (24, 128)
        indim = rows * cols
        odim = indim * nr

        # h_frags has the data in the order we expect on the device
        h_frags_T = np.random.randn(odim).reshape(
            (nr * cols, rows)).astype(self.be.default_dtype)
        h_frags = h_frags_T.transpose().astype(self.be.default_dtype, order='C')

        d_frags = self.be.empty((rows, cols))
        d_frags_T = self.be.empty((cols, rows))

        d_reps = self.be.empty((rows, cols * nr))
        d_reps_T = self.be.empty((cols * nr, rows))

        self.be.scatter(h_frags_T, d_frags_T)
        self.be.transpose(d_frags_T, d_frags)

        np.testing.assert_allclose(d_frags.asnumpyarray(),
                                   h_frags, atol=1e-5, rtol=0)

        self.be.fragment_to_replica(d_frags_T, d_reps_T)
        self.be.transpose(d_reps_T, d_reps)

        for i in range(nr):
            np.testing.assert_allclose(d_frags.asnumpyarray(),
                                       d_reps.tlist[i].asnumpyarray(),
                                       atol=1e-5, rtol=0)
        print("Frag2Rep OK")

        d_frags_T.fill(0)
        self.be.replica_to_fragment(d_reps_T, d_frags_T)
        self.be.transpose(d_frags_T, d_frags)
        for i in range(nr):
            np.testing.assert_allclose(d_frags.asnumpyarray(),
                                       d_reps.tlist[i].asnumpyarray(),
                                       atol=1e-5, rtol=0)
        print("Rep2Frag OK")

    @attr('fprop_shard')
    def fprop_test(self):
        nr = self.be.num_dev
        # create a numpy array as the test-bed
        indim = 1024
        odim = indim * nr
        mbsz = 64 * nr
        wtsz = (odim, indim)
        np.random.seed(0)
        np.set_printoptions(precision=4, linewidth=100)
        h_w = np.random.randn(wtsz[0] * wtsz[1]).reshape(
                wtsz).astype(self.be.default_dtype)
        h_d_T = np.random.randn(indim * mbsz).reshape(
                (mbsz, indim)).astype(self.be.default_dtype)

        h_d = h_d_T.transpose().astype(self.be.default_dtype, order='C')
        h_result = np.dot(h_w, h_d)

        d_w = self.be.empty((indim, indim))
        d_d = self.be.allocate_fragment((indim, mbsz))
        d_d_T = self.be.empty(d_d.shape[::-1])
        d_ubuf = self.be.allocate_fragment((indim, mbsz))
        d_obuf = self.be.allocate_fragment((odim, mbsz))

        d_o = self.be.allocate_fragment((odim, mbsz))

        self.be.scatter(h_d_T, d_d_T)
        self.be.transpose(d_d_T, d_d)
        self.be.scatter(h_w, d_w)

        self.be.fprop_fc_shard(d_o, d_d, d_w, layer=[d_ubuf, d_obuf])

        np.testing.assert_allclose(d_o.asnumpyarray(),
                                   h_result, atol=1e-3, rtol=0)
