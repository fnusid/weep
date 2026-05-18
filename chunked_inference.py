import pathlib
from pyexpat import model

import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dpcnn import DPCCN #TSE
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper 
from omegaconf import OmegaConf
from pathlib import Path
import yaml

TSE_CKPT = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt"
EMB_CKPT = "/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
joint_trained_ckpt = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"
config_path = Path("/home/sidcs/codebase/wesep/confs/config_dpcnn.yaml")

with config_path.open("r", encoding="utf-8") as f:
    docs = [OmegaConf.create(d) for d in yaml.safe_load_all(f)]

hp = OmegaConf.merge(*docs)

def load_dpccn_ckpt(model, ckpt_path):

    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd = ckpt["state_dict"]

    dpccn_sd = {}

    for k, v in sd.items():

        if k.startswith("model."):

            new_k = k[len("model."):]

            dpccn_sd[new_k] = v

    model.load_state_dict(dpccn_sd, strict=True)
    return model


def joint_trained_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith('dual_emb_model.'):
            k2 = k.replace('dual_emb_model.', '')
            new_state[k2] = v
    return new_state

def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        k2 = k.replace("model.", "")
        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue
        new_state[k2] = v
    return new_state


def load_dual_model(ckpt_path, emb_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = strip_dual_model_weights(ckpt["state_dict"])
    model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

def load_dual(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = strip_dual_model_weights(state_dict)
    model.load_state_dict(state_dict, strict=True)
    state_dict_joint_trained = torch.load(joint_trained_ckpt, map_location="cpu")["state_dict"]
    state_dict_joint_trained = joint_trained_model_weights(state_dict_joint_trained)
    model.load_state_dict(state_dict_joint_trained, strict=True)
    model.eval()
    
    return model

def load_audio(path, sample_rate=16_000):
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        wav = resampler(wav)
        sr = sample_rate
    
    return wav, sr


def chunked_inference(inference_wrapper, audio, chunk_ms=2000, hop_ms=1000, sample_rate=16_000):
    '''
    I am using Hann window here as synthesis, and rectangular as analysis, 
    so overlap should be 50% (hop_ms = chunk_ms / 2) to avoid artifacts.
    '''
    chunk_size = int(sample_rate * chunk_ms / 1000)
    hop_size = int(sample_rate * hop_ms / 1000)

    #estimating the number of chunks
    num_chunks = int(torch.ceil(torch.tensor((audio.shape[1] - chunk_size) / hop_size)).item()) + 1
    if num_chunks <= 0:
        raise ValueError("Audio is too short for the given chunk size and hop size.")
    total_len = (num_chunks - 1) * hop_size + chunk_size
    if total_len > audio.shape[1]:
        audio = F.pad(audio, (0, total_len - audio.shape[1]))
    # Prepare a tensor to hold the output
    output = torch.zeros((2, audio.shape[1]), dtype=audio.dtype)
    weight = torch.zeros((audio.shape[1],), dtype=audio.dtype, device=audio.device)

    window = torch.hann_window(chunk_size, periodic=True, device=audio.device)

    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = audio[:, start:end]

        #Run the model on the chunk
        with torch.no_grad():
            chunk_out = inference_wrapper.out_computer(chunk)
            
            output[:, start:end] += chunk_out * window
            weight[start:end] += window

    final_output = output / weight.clamp(min=1e-8).unsqueeze(0)
    return final_output
        

class ChunkedInferenceWrapper(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.tse_model = DPCCN(**hp.model_args.tse_model)   
        self.emb_model = SpeakerEncoderDualWrapper(emb_dim=256)

        self.tse_model = load_dpccn_ckpt(self.tse_model, TSE_CKPT)
        self.emb_model = load_dual(self.emb_model, EMB_CKPT)

        self.tse_model.eval()
        self.emb_model.eval()

        self.embedding1 = torch.zeros(1, 256) 
        self.embedding2 = torch.zeros(1,256)

    

    def update_embeddings(self, e1, e2):
        if F.cosine_similarity(e1, e2) >= 0.75:
            return

        cosine_e11 = F.cosine_similarity(e1, self.embedding1)
        cosine_e12 = F.cosine_similarity(e1, self.embedding2)
        cosine_e21 = F.cosine_similarity(e2, self.embedding1)
        cosine_e22 = F.cosine_similarity(e2, self.embedding2)

        if cosine_e11 > cosine_e12 and cosine_e22 > cosine_e21:
            self.embedding1 = self.embedding1 * 0.9 + e1 * 0.1
            self.embedding2 = self.embedding2 * 0.9 + e2 * 0.1
        elif cosine_e12 > cosine_e11 and cosine_e21 > cosine_e22:
            self.embedding1 = self.embedding1 * 0.9 + e2 * 0.1
            self.embedding2 = self.embedding2 * 0.9 + e1 * 0.1
        else:
            pass

    def out_computer(self, chunk):
        embs = self.emb_model.forward(chunk)
        #Normalize the embeddings
        
        e1, e2 = embs[0][0].unsqueeze(0), embs[0][1].unsqueeze(0)
        # breakpoint()
        self.update_embeddings(e1, e2)
        # self.embedding1 = F.normalize(self.embedding1, dim=-1)
        # self.embedding2 = F.normalize(self.embedding2, dim=-1)

        out1,_ = self.tse_model.forward(chunk, self.embedding1)
        out2,_ = self.tse_model.forward(chunk, self.embedding2)
      
        out = torch.cat([out1, out2], dim=0)
        return out



    def forward(self, wav_path):
        wav, sr = load_audio(wav_path)

        ## True offline evaluation
        # embs = self.emb_model.forward(wav)
        # e1, e2 = embs[0]
        # out1,_ = self.tse_model.forward(wav, e1.unsqueeze(0))
        # out2,_ = self.tse_model.forward(wav, e2.unsqueeze(0))
        # out = torch.cat([out1, out2], dim=0)
        # torchaudio.save('/home/sidcs/codebase/wesep/analysis/output1_full.wav', out[0].unsqueeze(0), sample_rate=sr)
        # torchaudio.save('/home/sidcs/codebase/wesep/analysis/output2_full.wav', out[1].unsqueeze(0), sample_rate=sr)
        # breakpoint()

        #wait 6s initially for reliable embedding extraction, then do chunked inference
        initial_chunk_size = int(sr * wav.shape[-1] / sr)
        initial_chunk = wav[:, :initial_chunk_size]
        breakpoint()
        with torch.no_grad():
            emb = self.emb_model.forward(initial_chunk)
            #Normalize the embeddings
            # emb = F.normalize(emb, dim=-1)
            e1, e2 = emb.squeeze(0)
            if e1.ndim == 1:
                e1 = e1.unsqueeze(0)
            if e2.ndim == 1:
                e2 = e2.unsqueeze(0)
            self.embedding1 = e1
            self.embedding2 = e2
        
        # final_output = chunked_inference(self, wav[:, initial_chunk_size:], chunk_ms=2000, hop_ms=1000, sample_rate=sr)
        # final_output = chunked_inference(self, wav, chunk_ms=2000, hop_ms=1000, sample_rate=sr)
        final_output = chunked_inference(self, wav, chunk_ms=wav.shape[-1]/16000*1000, hop_ms=(wav.shape[-1]/16000*1000)/2, sample_rate=sr)
        #Normalize the final output audio
        breakpoint()
        # final_output = final_output / final_output.abs().max().clamp(min=1e-8)

        torchaudio.save('/home/sidcs/codebase/wesep/analysis/out1_61-70968-0005_5105-28233-0000_chunked_inference.wav', final_output[0].unsqueeze(0), sample_rate=sr)
        torchaudio.save('/home/sidcs/codebase/wesep/analysis/out2_61-70968-0005_5105-28233-0000_chunked_inference.wav', final_output[1].unsqueeze(0), sample_rate=sr)

        
        
        
if __name__ == "__main__":
    inference_wrapper = ChunkedInferenceWrapper()
    inference_wrapper("/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/test/mix_both/61-70968-0005_5105-28233-0000.wav")










    