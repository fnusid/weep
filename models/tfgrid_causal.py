import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/sidcs/codebase/wesep')
from modules.gridnet_causal.film import FilmLayer


def get_layer(activation):
    if activation == 'prelu':
        return nn.PReLU()
    raise NotImplementedError

def ILD(x1, x2, tol = 1e-6):
    # x - B, T
    # ILD - B, F, T
    ILD = torch.log10(torch.div(x1.abs() + tol, x2.abs() + tol))
    return ILD

def IPD(x1, x2, tol = 1e-6):
    # x - B, T
    # ILD - B, F, T
    IPD =  torch.angle(x1) -  torch.angle(x2)
    IPD_cos = torch.cos(IPD)
    IPD_sin = torch.sin(IPD)
    IPD_map = torch.cat((IPD_sin, IPD_cos), dim = 1)
    return IPD_map


def IPD_ONNX(real1, imag1, real2, imag2, norm, norm_ref, tol = 1e-6):
    B, _, f, T = real2.shape

    real2 = real2.repeat((1, real1.shape[1], 1, 1))#.reshape(B*(M-1), 1, f, T)
    imag2 = imag2.repeat((1, imag1.shape[1], 1, 1))#.reshape(B*(M-1), 1, f, T)

    IPD_cos = (real1 * real2 + imag1 * imag2) / (norm * norm_ref + tol)
    IPD_sin = (real2 * imag1 - imag2 * real1) / (norm * norm_ref + tol)
    
    IPD_cos = IPD_cos.reshape(-1, 1, f, T)
    IPD_sin = IPD_sin.reshape(-1, 1, f, T)
    
    IPD_map = torch.cat((IPD_sin, IPD_cos), dim = 1)

    IPD_map = IPD_map.reshape(B, 2 * imag1.shape[1], f, T)

    return IPD_map


def MC_features_ONNX(reals, imags, eps=1e-6):
    r2, r1 = torch.split(reals, [1, reals.shape[1] - 1], dim=1)
    i2, i1 = torch.split(imags, [1, reals.shape[1] - 1], dim=1)
    
    # Compute magnitude
    norm = torch.sqrt(torch.square(reals) + torch.square(imags))
    norm_ref, norm = torch.split(norm, [1, norm.shape[1] - 1], dim=1)

    # Compute ILD
    ILD_m = torch.log10(torch.div(norm + eps, norm_ref + eps))

    # Compute IPD
    IPD_m = IPD_ONNX(r1, i1, r2, i2, norm, norm_ref)

    out = torch.cat([ILD_m, IPD_m], dim=1) # [B, 3M-3, f, T]
    
    return out

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1) # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # [B, C, T, F]
        return x

class TFGridNet(nn.Module):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNet blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNet model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        spk_emb_dim=256,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        lstm_down=4,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=1,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        ref_channel=-1,
        use_attn=False,
        chunk_causal=True,
        local_atten_len=100,
        spectral_masking=True,
        use_first_ln=False,
        merge_method = "None",
        conv_lstm = True,
        masked_attn = False
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.ref_channel = ref_channel
        self.emb_dim = emb_dim
        self.eps = eps
        self.chunk_size = stride
        self.spectral_masking = spectral_masking
        self.merge_method = merge_method

        self.istft_pad = n_fft - stride

        self.lookahead = self.istft_pad

        # ISTFT overlap-add will affect this many chunks in the future
        self.istft_lookback = 1 + (self.istft_pad - 1) // self.istft_pad
        
        self.n_fft = n_fft
        self.window = window

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, 3), (0, 1)
        
        Feat_num = (n_imics - 1)*3 

        self.Feat_num = Feat_num
        if self.merge_method == "None":
            module_list = [nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding)]
        elif self.merge_method == "early_cat":
            self.emb_dim = emb_dim
            module_list = [
                nn.Conv2d(2 * n_imics + Feat_num, emb_dim, ks, padding=padding),
            ]

        if use_first_ln:
            module_list.append(LayerNormPermuted(emb_dim))
        
        self.conv = nn.Sequential(
            *module_list
        )
        
        self.embedding_modules = nn.ModuleList([])

        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            
            self.embedding_modules.append(FilmLayer(spk_emb_dim, self.emb_dim))
            
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    lstm_down,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    use_attn=use_attn,
                    chunk_causal=chunk_causal,
                    local_atten_len=local_atten_len,
                    conv_lstm = conv_lstm,
                    masked_attn = masked_attn
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=( self.t_ksize - 1, 1))
    
    def init_buffers(self, batch_size, device):
        if self.merge_method == "None": 
            conv_buf = torch.zeros(batch_size, self.n_imics*2, self.t_ksize - 1, self.n_freqs,
                                device=device)
        elif self.merge_method == "early_cat":
            conv_buf = torch.zeros(batch_size, self.n_imics*2+self.Feat_num, self.t_ksize - 1, self.n_freqs,
                    device=device)
            
        deconv_buf = torch.zeros(batch_size, self.emb_dim, self.t_ksize - 1, self.n_freqs,
                                 device=device)

        gridnet_buffers = {}
        for i in range(len(self.blocks)):
            gridnet_buffers[f'buf{i}'] = self.blocks[i].init_buffers(batch_size, device)

        return dict(conv_buf=conv_buf, deconv_buf=deconv_buf,
                    gridnet_bufs=gridnet_buffers)

    def forward(self, input_stft: torch.Tensor, embedding: torch.Tensor, input_state) -> torch.Tensor:
        """
        B: batch, M: mic, F: real/imag, C: freq bin, T: time frame
        input_stft: (B, M, C*F, T)
        embedding: (B, D)
        output: (B, M, C*F, T)
        """

        if input_state is None:
            input_state = self.init_buffers(input_stft.shape[0], input_stft.device)
        
        conv_buf = input_state['conv_buf']
        deconv_buf = input_state['deconv_buf']
        gridnet_buf = input_state['gridnet_bufs']
        
        batch = input_stft
        
        real, imag = torch.split(batch, [self.n_freqs, self.n_freqs], dim=-2)
        
        batch = torch.cat((real, imag), dim=1)  # [B, 2*M, F, T]

        if self.merge_method == "None":
            batch = batch.transpose(2, 3) # [B, M, T, F]
            n_batch, _, n_frames, n_freqs = batch.shape # B, 2M, T, F
            batch = torch.cat((conv_buf, batch), dim=2)
            conv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
            batch = self.conv(batch)  # [B, -1, T, F]
        else:
            Feats = MC_features_ONNX(real, imag)
            batch = torch.cat((batch, Feats), dim=1)
            batch = batch.transpose(2, 3) # [B, M, T, F]
            n_batch, _, n_frames, n_freqs = batch.shape # B, 2M, T, F
            
            batch = torch.cat(( conv_buf, batch), dim=2)
            conv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
            
            batch = self.conv(batch)  # [B, -1, T, F]

        # BCTQ
        batch = batch.permute(0, 2, 3, 1)
        embedding = embedding.unsqueeze(1).unsqueeze(1)
        
        for ii in range(self.n_layers):
            batch = self.embedding_modules[ii](batch, embedding)
            
            batch, gridnet_buf[f'buf{ii}'] = self.blocks[ii](batch, gridnet_buf[f'buf{ii}']) # [B, T, Q, C]

        batch = batch.permute(0, 3, 1, 2) # [B, C, T, Q]
        
        batch = torch.cat(( deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :,  -(self.t_ksize - 1):, :]
        
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]batch ] 
        batch = batch[:, :, -n_frames:, :n_freqs]
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs]) # [B, n_srcs, 2, n_frames, n_freqs]
        
        batch = batch.transpose(3, 4) # (B, n_srcs, 2, n_fft//2 + 1, T)
        
        # Concat real and imaginary parts
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2) # (B, n_srcs, nfft + 2, T)

        # Do spectral masking
        if self.spectral_masking:
            batch = batch * input_stft[:, :self.n_srcs] # First few channels only

        input_state['conv_buf'] = conv_buf
        input_state['deconv_buf'] = deconv_buf
        input_state['gridnet_bufs'] = gridnet_buf

        return batch, input_state


class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        lstm_down,
        n_head=4,
        local_atten_len= 100,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        use_attn=True,
        chunk_causal = True,
        conv_lstm = True,
        masked_attn = False
    ):
        super().__init__()
        bidirectional = False # Causal
        self.use_attn = use_attn
        self.local_atten_len = local_atten_len
        self.E = math.ceil(
                    approx_qk_dim * 1.0 / n_freqs
                )  # approx_qk_dim is only approximate
        self.masked_attn = masked_attn
        self.n_head = n_head if use_attn else 1
        self.V_dim = emb_dim // self.n_head
        self.chunk_causal = chunk_causal
        self.H = hidden_channels
        self.lstm_down = lstm_down
        
        in_channels = emb_dim
        self.in_channels = in_channels
        self.n_freqs = n_freqs

        ## intra RNN can be optimized by conv or linear because the frequence length are not very large
        self.conv_lstm = conv_lstm
        if conv_lstm:
            self.conv = nn.Conv1d(in_channels=emb_dim,
                                out_channels=emb_dim,
                                kernel_size=lstm_down,
                                stride=lstm_down)
            self.act = nn.PReLU()
            self.norm = LayerNormalization4D(emb_dim)
            
            self.intra_rnn = nn.LSTM(
                in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
            )
            
            self.deconv = nn.ConvTranspose1d(in_channels=hidden_channels * 2,
                                            out_channels=emb_dim,
                                            kernel_size=lstm_down,
                                            stride=lstm_down,
                                            output_padding=n_freqs - (n_freqs//lstm_down) * lstm_down)
        else:
            self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
            self.intra_rnn = nn.LSTM(
                in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
            )
            self.intra_linear = nn.Linear(
                hidden_channels*2, emb_dim,
            )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM( 
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=bidirectional,
        )
        self.inter_linear = nn.Linear(
            hidden_channels*(bidirectional + 1), emb_dim
        )
        
        if self.use_attn:
            E = self.E
            assert emb_dim % n_head == 0
            self.add_module(
                "attn_conv_Q",
                nn.Sequential(
                    nn.Linear(emb_dim, E * n_head), # [B, T, Q, HE]
                    get_layer(activation),
                    # [B, T, Q, H, E] -> [B, H, T, Q, E] ->  [B * H, T, Q * E]
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                      .permute(0, 3, 1, 2, 4)\
                                      .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)), # (BH, T, Q * E)
                    LayerNormalization4DCF((n_freqs, E), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K",
                nn.Sequential(
                    nn.Linear(emb_dim, E * n_head),
                    get_layer(activation),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                      .permute(0, 3, 1, 2, 4)\
                                      .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)),
                    LayerNormalization4DCF((n_freqs, E), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V",
                nn.Sequential(
                    nn.Linear(emb_dim, (emb_dim // n_head) * n_head),
                    get_layer(activation),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, (emb_dim // n_head))\
                                      .permute(0, 3, 1, 2, 4)\
                                      .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * (emb_dim // n_head))),
                    LayerNormalization4DCF((n_freqs, emb_dim // n_head), eps=eps),
                ),
            )
            self.add_module(
                "attn_concat_proj",
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    get_layer(activation),
                    LayerNormalization4D(emb_dim, eps=eps),
                ),
            )
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def _init_buffers(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.in_channels, self.local_atten_len - 1, self.n_freqs),
            device=device)
    
    def init_buffers(self, batch_size, device):
        ctx_buf = {}
        
        if self.use_attn:
            K_buf = torch.zeros((batch_size * self.n_head,
                                self.local_atten_len - 1,
                                self.E * self.n_freqs), device=device)
            ctx_buf['K_buf'] = K_buf
            
            V_buf = torch.zeros((batch_size * self.n_head,
                                self.local_atten_len - 1,
                                self.V_dim * self.n_freqs), device=device)
            ctx_buf['V_buf'] = V_buf
            
        c0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        ctx_buf['c0'] = c0

        h0 = torch.zeros((1,
                          batch_size * self.n_freqs,
                          self.H), device=device)
        ctx_buf['h0'] = h0

        return ctx_buf

    def _causal_unfold_chunk(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.

        Args:
            x: [B, T, QC], L is total length of signal
            ctx_len: int
        Returns:
            [B * num_chunk, QC, atten_len]
        """
        x = x.transpose(1, 2) # [B, QC, T]
        
        if x.shape[-1] == self.local_atten_len:
            return x
        
        # print('A', x.shape)
        x = x.unfold(2, self.local_atten_len, 1) # [B, QC, num_chunk, atten_len]
        
        B, QC, N, L = x.shape
        x = x.transpose(1, 2).reshape(B * N, QC, L)

        return x
        
    def get_lookahead_mask(self, query_len, key_len, device):
        """Return a causal local-attention mask with optional left-context truncation."""
        query_positions = torch.arange(query_len, device=device).unsqueeze(1)
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        future_ok = key_positions <= query_positions + (key_len - query_len)
        history_ok = key_positions >= query_positions + (key_len - query_len) - (self.local_atten_len - 1)
        return (future_ok & history_ok).detach()

    def forward(self, x, init_state = None, debug=False):
        """GridNetBlock Forward.

        Args:
            x: [B, T, Q, C]
            out: [B, T, Q, C]
        """
        
        if init_state is None:
            init_state = self.init_buffers(x.shape[0], x.device)

        B, T, Q, C = x.shape
        old_T = T
        
        # intra RNN
        input_ = x

        if self.conv_lstm:
            intra_rnn = input_.reshape(B * T, Q, C)  # [B * T, Q, C]
            
            intra_rnn = self.conv(intra_rnn.transpose(1, 2)) # [BT, C, K] K = Q // stride
            intra_rnn = self.act(intra_rnn)
            intra_rnn = self.norm(intra_rnn.transpose(1, 2)) # [BT, K, C]
            
            self.intra_rnn.flatten_parameters()
            
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
            
            intra_rnn = self.deconv(intra_rnn.transpose(1, 2)) # [BT, C, Q]

            intra_rnn = intra_rnn.transpose(1, 2)
        else:
            intra_rnn = self.intra_norm(input_) # [B, T, Q, C]
            intra_rnn = intra_rnn.reshape(B * T, Q, C)  # [B * T, Q, C]
            self.intra_rnn.flatten_parameters()

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q, C]
        
        intra_rnn = intra_rnn.view(B, T, Q, C) # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]
        out = intra_rnn

        # inter RNN
        input_ = intra_rnn # [B, T, Q, C]
        
        inter_rnn = self.inter_norm(intra_rnn)  # [B, T, Q, C]
        inter_rnn = inter_rnn.transpose(1, 2).reshape(B * Q, T, C)  # [BQ, T, C]
        
        self.inter_rnn.flatten_parameters()
        
        h0 = init_state['h0']
        c0 = init_state['c0']

        inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # [BQ, -1, H]
       
        init_state['h0'] = h0
        init_state['c0'] = c0
       
        inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T, C]
        
        inter_rnn = inter_rnn.view([B, Q, T, C])
        inter_rnn = inter_rnn.transpose(1, 2) # [B, T, Q, C]
        inter_rnn = inter_rnn + input_  # [B, T, Q, C]
        
        out = inter_rnn


        # Attn
        if self.use_attn:
            if self.masked_attn:
                batch = inter_rnn
                Q = self["attn_conv_Q"](batch)  # [B*H, T, Q*E]
                K_cur = self["attn_conv_K"](batch)
                V_cur = self["attn_conv_V"](batch)

                K_buf = init_state['K_buf']
                V_buf = init_state['V_buf']

                K = torch.cat([K_buf, K_cur], dim=1)
                V = torch.cat([V_buf, V_cur], dim=1)

                init_state['K_buf'] = K[:, -(self.local_atten_len - 1):, :]
                init_state['V_buf'] = V[:, -(self.local_atten_len - 1):, :]

                local_mask = self.get_lookahead_mask(Q.shape[1], K.shape[1], inter_rnn.device)

                emb_dim = Q.shape[-1]
                attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim ** 0.5)
                attn_mat = attn_mat.masked_fill(~local_mask.unsqueeze(0), -float('inf'))
                attn_mat = F.softmax(attn_mat, dim=2)
                V = torch.matmul(attn_mat, V)
                V = V.reshape(B, self.n_head, old_T, self.n_freqs, self.V_dim)
                V = V.permute(0, 2, 3, 1, 4).reshape(
                    B, old_T, self.n_freqs, self.n_head * self.V_dim
                )
                batch = self["attn_concat_proj"](V)

                out = out + batch
            else:
                # Attention by unfolding (useful for streaming inference)
                # attention
                batch = inter_rnn

                Q = self["attn_conv_Q"](batch)
                K = self["attn_conv_K"](batch)
                V = self["attn_conv_V"](batch)

                # Buffers
                K_buf = init_state['K_buf']
                V_buf = init_state['V_buf']

                K = torch.cat([K_buf, K], dim=1)
                V = torch.cat([V_buf, V], dim=1)
                init_state['K_buf'] = K[:, -(self.local_atten_len - 1):, :]
                init_state['V_buf'] = V[:, -(self.local_atten_len - 1):, :]

                Q = Q.reshape(Q.shape[0] * Q.shape[1], 1, Q.shape[2])
                K = self._causal_unfold_chunk(K)
                V = self._causal_unfold_chunk(V)

                emb_dim = Q.shape[-1]
                attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)
                attn_mat = F.softmax(attn_mat, dim=2)

                V = torch.matmul(attn_mat, V.transpose(1, 2))
                V = V.reshape(B, self.n_head, old_T, self.n_freqs, self.V_dim)
                V = V.permute(0, 2, 3, 1, 4).reshape(
                    B, old_T, self.n_freqs, self.n_head * self.V_dim
                )
                batch = self["attn_concat_proj"](V)

                out = out + batch

        return out, init_state


# Use native layernorm implementation
class LayerNormalization4D(nn.Module):
    def __init__(self, C, eps=1e-5, preserve_outdim=False):
        super().__init__()
        self.norm = nn.LayerNorm(C, eps=eps)
        self.preserve_outdim = preserve_outdim

    def forward(self, x: torch.Tensor):
        """
        input: (*, C)
        """
        x = self.norm(x)
        return x
    
class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        assert len(input_dimension) == 2
        Q, C = input_dimension
        super().__init__()
        self.norm = nn.LayerNorm((Q * C), eps=eps)

    def forward(self, x: torch.Tensor):
        """
        input: (B, T, Q * C)
        """
        x = self.norm(x)

        return x



if __name__ == "__main__":
    r1 = torch.rand(4, 1, 48000)*10
    r2 = torch.rand(4, 1, 48000)*10
    i1 = torch.rand(4, 1, 48000)*10
    i2 = torch.rand(4, 1, 48000)*10

    c1 = torch.complex(r1, i1)
    c2 = torch.complex(r2, i2)

    ILD1 = ILD(c1, c2)
    ILD2 = ILD_OMNX(r1, i1, r2, i2)

    IPD2 = IPD_OMNX(r1, i1, r2, i2)
    IPD1 = IPD(c1, c2)


    print(torch.allclose(ILD1, ILD2, atol = 1e-2)  )
    print(torch.allclose(IPD1, IPD2, atol = 1e-2)  )