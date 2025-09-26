import torch
from torch import nn
from layers.Embed import PatchEmbedding
from einops import rearrange
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from typing import Tuple
import torch.nn.functional as F



class AdaptiveLocalExpertSelection(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, m, c) tensor
        Return:
            r_weight, topk_index: (n, m, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, m, m)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, m, k), (n, m, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, m, k)
        
        return r_weight, topk_index
        

class ExpertGathering(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, m, topk) tensor
        r_weight: (n, m, topk) tensor
        kv: (n, m, c_kq+c_v)

        Return:
            (n, m, topk, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, m, c_kv = kv.size()
        topk = r_idx.size(-1)

        all_kv = kv.unsqueeze(1).expand(-1, m, -1, -1)
        r_idx_expanded = r_idx.unsqueeze(-1).expand(-1, -1, -1, c_kv)
        topk_kv = torch.gather(all_kv, dim=2, index=r_idx_expanded)
        
        


        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, m, topk, 1) * topk_kv # (n, m, k, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        return q, k, v

class TMOE(nn.Module):
    def __init__(self, dim, num_heads=8, topk=4, shared=False, qk_dim=None, qk_scale=None, mul_weight='none'):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim or dim
        self.num_heads = num_heads
        self.topk = topk
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.pos = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        
        if topk > 0:
            self.router = AdaptiveLocalExpertSelection(qk_dim=self.qk_dim,
                                    qk_scale=self.scale,
                                    topk=self.topk,
                                    )
            self.kv_gather = ExpertGathering(mul_weight = mul_weight)
        self.qkv = QKVLinear(self.dim, self.qk_dim)
        self.wo = nn.Linear(dim, dim)
        self.attn_act = nn.Softmax(dim=-1)
        self.shared = shared
        if shared:
            self.state_linear = nn.Linear(dim, self.qk_dim)
    
    def forward(self, x):
        """
        x: NMC tensor

        Return:
            NMC tensor
        """
        x = x + self.pos(x.permute(0, 2, 1)).permute(0, 2, 1)
        n, m, c = x.shape
        q, k, v = self.qkv(x)
        q = rearrange(q, 'n m (h d)-> (n h) m d', h=self.num_heads)
        k = rearrange(k, 'n m (h d)-> (n h) m d', h=self.num_heads)
        v = rearrange(v, 'n m (h d)-> (n h) m d', h=self.num_heads)
        if self.shared:
            # state = self.state_linear(x)
            state = x
            state = torch.mean(state, dim=1, keepdim=True)

        d = int(c / self.num_heads)
        if self.topk > 0:
            r_weight, r_idx = self.router(q, k) # both are (n, m, topk) tensors
            kv = torch.cat([k, v], dim=-1)
            kv_select = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv)
            k_select, v_select = kv_select.split([int(self.qk_dim / self.num_heads), int(self.dim/self.num_heads)], dim=-1) # (n, m, k, d)
        else:
            k_select, v_select = k, v
        if self.shared:
            if self.topk > 0:
                state = state.repeat(1, m, 1)
                state = rearrange(state, 'n m (h d)-> (n h) m d', h=self.num_heads)
                k_select = torch.cat([k_select, state.unsqueeze(2)], dim=2) #(n,m,k+1, d)
                v_select = torch.cat([v_select, state.unsqueeze(2)], dim=2) #(n,m,k+1, d)
            else:
                state = rearrange(state, 'n 1 (h d)-> (n h) 1 d', h=self.num_heads)
                k_select = torch.cat([k_select, state], dim=1) #(n,m+1, d)
                v_select = torch.cat([v_select, state], dim=1) #(n,m+1, d)

        # TMOE Context Aggregation)
        if self.topk>0:
            attn_weight = (q * self.scale).view(n*self.num_heads, m, 1, d) @ k_select.permute(0, 1, 3, 2) #(n, m, 1, k)
            attn_weight = self.attn_act(attn_weight)
            out = attn_weight @ v_select #(n, m, 1, d)
            out = rearrange(out, '(n h) m 1 d-> n m (h d)', h=self.num_heads)
        else:
            attn_weight = (q * self.scale) @ k_select.permute(0, 2, 1)
            attn_weight = self.attn_act(attn_weight)
            out = attn_weight @ v_select #(n, m, d)
            out = rearrange(out, '(n h) m d-> n m (h d)', h=self.num_heads)
        out = self.wo(out)
        return out

class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TMOELayer(nn.Module):
    def __init__(self, dim, num_heads, topk, shared, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.attn = TMOE(dim, num_heads, topk, shared)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.drop_path(self.mlp(x))
        return x

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Model(nn.Module):

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        patch_len = configs.patch_len
        stride = configs.stride
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                TMOELayer(
                    configs.d_model,
                    configs.n_heads,
                    configs.topk,
                    configs.shared
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None