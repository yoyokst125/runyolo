import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class RK4Optimizer(Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate/step size: {lr}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # 用於記錄梯度變化（曲率信息）
        self.gradient_diffs = []

    @torch.no_grad()
    def step(self, closure):
        loss = None

        # 收集參數和狀態
        params_with_state = []
        for group in self.param_groups:
            h_group = group['lr']
            wd_group = group['weight_decay']
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

        def get_grads_after_closure_and_clear_previous(apply_wd=False):
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
            for i, p_data in enumerate(params_with_state):
                p_data['param'].data.copy_(p_data['original_state'] + delta_thetas_from_original[i])

        def reset_params_to_original_state():
            for p_data in params_with_state:
                p_data['param'].data.copy_(p_data['original_state'])

        # RK4 計算
        reset_params_to_original_state()
        loss, g_k1 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k1_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k1)]

        temp_delta_thetas_for_k2 = [k1_d / 2.0 for k1_d in k1_delta_thetas]
        set_params_to_intermediate_state(temp_delta_thetas_for_k2)
        _, g_k2 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k2_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k2)]

        temp_delta_thetas_for_k3 = [k2_d / 2.0 for k2_d in k2_delta_thetas]
        set_params_to_intermediate_state(temp_delta_thetas_for_k3)
        _, g_k3 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k3_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k3)]

        temp_delta_thetas_for_k4 = k3_delta_thetas
        set_params_to_intermediate_state(temp_delta_thetas_for_k4)
        _, g_k4 = get_grads_after_closure_and_clear_previous(apply_wd=True)
        k4_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, g_k4)]

        # 記錄梯度差異（近似曲率）
        grad_diff_k2_k1 = [torch.norm(g_k2[i] - g_k1[i]).item() for i in range(len(g_k1))]
        grad_diff_k3_k2 = [torch.norm(g_k3[i] - g_k2[i]).item() for i in range(len(g_k2))]
        grad_diff_k4_k3 = [torch.norm(g_k4[i] - g_k3[i]).item() for i in range(len(g_k3))]
        self.gradient_diffs.append({
            'k2_k1': grad_diff_k2_k1,
            'k3_k2': grad_diff_k3_k2,
            'k4_k3': grad_diff_k4_k3
        })

        # 最終參數更新
        for i, p_data in enumerate(params_with_state):
            final_combined_delta = (k1_delta_thetas[i] +
                                    2 * k2_delta_thetas[i] +
                                    2 * k3_delta_thetas[i] +
                                    k4_delta_thetas[i]) / 6.0
            p_data['param'].data.copy_(p_data['original_state'] + final_combined_delta)

        return loss
