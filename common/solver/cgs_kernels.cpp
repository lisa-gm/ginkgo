/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/cgs_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The CGS solver namespace.
 *
 * @ingroup cgs
 */
namespace cgs {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r_tld, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *u,
                matrix::Dense<ValueType> *u_hat,
                matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *beta,
                matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho,
                Array<stopping_status> *stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto r_tld, auto p,
                      auto q, auto u, auto u_hat, auto v_hat, auto t,
                      auto alpha, auto beta, auto gamma, auto prev_rho,
                      auto rho, auto stop) {
            if (row == 0) {
                rho[col] = zero(rho[col]);
                prev_rho[col] = alpha[col] = beta[col] = gamma[col] =
                    one(prev_rho[col]);
                stop[col].reset();
            }
            r(row, col) = r_tld(row, col) = b(row, col);
            u(row, col) = u_hat(row, col) = p(row, col) = q(row, col) =
                v_hat(row, col) = t(row, col) = zero(u(row, col));
        },
        b->get_size(), b->get_stride(), default_stride(b), default_stride(r),
        default_stride(r_tld), default_stride(p), default_stride(q),
        default_stride(u), default_stride(u_hat), default_stride(v_hat),
        default_stride(t), row_vector(alpha), row_vector(beta),
        row_vector(gamma), row_vector(prev_rho), row_vector(rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const Array<stopping_status> *stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto r, auto u, auto p, auto q,
                      auto beta, auto rho, auto prev_rho, auto stop) {
            if (!stop[col].has_stopped()) {
                auto prev_rho_zero = prev_rho[col] == zero(prev_rho[col]);
                auto tmp = prev_rho_zero ? beta[col] : rho[col] / prev_rho[col];
                if (row == 0 && !prev_rho_zero) {
                    beta[col] = tmp;
                }
                u(row, col) = r(row, col) + tmp * q(row, col);
                p(row, col) =
                    u(row, col) + tmp * (q(row, col) + tmp * p(row, col));
            }
        },
        r->get_size(), r->get_stride(), default_stride(r), default_stride(u),
        default_stride(p), default_stride(q), row_vector(beta), row_vector(rho),
        row_vector(prev_rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *t, matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *gamma,
            const Array<stopping_status> *stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto u, auto v_hat, auto q, auto t,
                      auto alpha, auto rho, auto gamma, auto stop) {
            if (!stop[col].has_stopped()) {
                auto gamma_is_zero = gamma[col] == zero(gamma[col]);
                auto tmp = gamma_is_zero ? alpha[col] : rho[col] / gamma[col];
                if (row == 0 && !gamma_is_zero) {
                    alpha[col] = tmp;
                }
                q(row, col) = u(row, col) - tmp * v_hat(row, col);
                t(row, col) = u(row, col) + q(row, col);
            }
        },
        u->get_size(), u->get_stride(), default_stride(u),
        default_stride(v_hat), default_stride(q), default_stride(t),
        row_vector(alpha), row_vector(rho), row_vector(gamma), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);

template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *u_hat, matrix::Dense<ValueType> *r,
            matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *alpha,
            const Array<stopping_status> *stop_status)
{
    run_kernel_solver(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto t, auto u_hat, auto r, auto x,
                      auto alpha, auto stop) {
            if (!stop[col].has_stopped()) {
                x(row, col) += alpha[col] * u_hat(row, col);
                r(row, col) -= alpha[col] * t(row, col);
            }
        },
        t->get_size(), t->get_stride(), default_stride(t),
        default_stride(u_hat), default_stride(r), x, row_vector(alpha),
        *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
