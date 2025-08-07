# utils/custom_optimizers.py
import torch
from torch.optim.optimizer import Optimizer
import math

class RK4Optimizer(Optimizer):
    """
    Implements Butcher's fifth-order Runge-Kutta method.
    The update rule is based on the provided Butcher tableau.
    """
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate/step size: {lr}")
        defaults = dict(lr=lr, weight_decay_rk5=0.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        loss = None

        # <<< --- 步驟 1: 首先收集所有可訓練參數及其狀態 --- >>>
        # 這個列表需要在任何使用它的嵌套函数定義之前被完全構建好
        params_with_state = []
        for group in self.param_groups:
            h_group = group['lr']
            wd_group = group.get('weight_decay_rk5', 0.0)

            for p in group['params']:
                if p.requires_grad:
                    params_with_state.append({
                        'param': p,
                        'original_state': p.clone().detach(),
                        'h': h_group,
                        'wd': wd_group
                    })

        if not params_with_state:
            if closure is not None:
                loss = closure()
            return loss

        # <<< --- 步驟 2: 現在 params_with_state 已定義，可以定義使用它的嵌套函數 --- >>>
        def get_grads_after_closure_and_clear_previous(apply_wd=False):
            # 此函數會使用外部作用域的 params_with_state
            for p_data in params_with_state:
                p = p_data['param']
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            with torch.enable_grad():
                current_loss = closure()

            grads = []
            for p_data in params_with_state:
                p = p_data['param']
                grad = torch.zeros_like(p.data) if p.grad is None else p.grad.clone().detach()

                if apply_wd and p_data['wd'] != 0:
                    grad.add_(p.data, alpha=p_data['wd'])
                grads.append(grad)
            return current_loss, grads

        def set_params_to_intermediate_state(delta_thetas_from_original):
            # 此函數會使用外部作用域的 params_with_state
            for i, p_data in enumerate(params_with_state):
                p_data['param'].data.copy_(p_data['original_state'] + delta_thetas_from_original[i])

        def reset_params_to_original_state():
            # 此函數會使用外部作用域的 params_with_state
            for p_data in params_with_state:
                p_data['param'].data.copy_(p_data['original_state'])

        # --- Butcher's 5th-Order RK 計算開始 ---
        reset_params_to_original_state()

        # k1 計算
        loss, g_k1 = get_grads_after_closure_and_clear_previous(apply_wd=True) # loss 是在 theta_n 計算的
        k1_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k1)]

        # k2 計算
        temp_delta_thetas_for_k2 = [k1_d / 4.0 for k1_d in k1_delta_thetas]
        set_params_to_intermediate_state(temp_delta_thetas_for_k2)
        _, g_k2 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k2_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k2)]

        # k3 計算
        temp_delta_thetas_for_k3 = [(k1_d / 8.0) + (k2_d / 8.0) for k1_d, k2_d in zip(k1_delta_thetas, k2_delta_thetas)]
        set_params_to_intermediate_state(temp_delta_thetas_for_k3)
        _, g_k3 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k3_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k3)]

        # k4 計算
        temp_delta_thetas_for_k4 = [(-k2_d / 2.0) + k3_d for k2_d, k3_d in zip(k2_delta_thetas, k3_delta_thetas)]
        set_params_to_intermediate_state(temp_delta_thetas_for_k4)
        _, g_k4 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k4_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k4)]

        # k5 計算
        temp_delta_thetas_for_k5 = [(3.0 * k1_d / 16.0) + (9.0 * k4_d / 16.0) for k1_d, k4_d in zip(k1_delta_thetas, k4_delta_thetas)]
        set_params_to_intermediate_state(temp_delta_thetas_for_k5)
        _, g_k5 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k5_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k5)]

        # k6 計算
        temp_delta_thetas_for_k6 = [
            (-3.0 * k1_d / 7.0) + (2.0 * k2_d / 7.0) + (12.0 * k3_d / 7.0) - (12.0 * k4_d / 7.0) + (8.0 * k5_d / 7.0)
            for k1_d, k2_d, k3_d, k4_d, k5_d in zip(k1_delta_thetas, k2_delta_thetas, k3_delta_thetas, k4_delta_thetas, k5_delta_thetas)
        ]
        set_params_to_intermediate_state(temp_delta_thetas_for_k6)
        _, g_k6 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k6_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k6)]

        # 最終參數更新
        for i, p_data in enumerate(params_with_state):
            final_combined_delta = (7.0 * k1_delta_thetas[i] +
                                    32.0 * k3_delta_thetas[i] +
                                    12.0 * k4_delta_thetas[i] +
                                    32.0 * k5_delta_thetas[i] +
                                    7.0 * k6_delta_thetas[i]) / 90.0
            p_data['param'].data.copy_(p_data['original_state'] + final_combined_delta)

        return loss
