import torch
import torch.nn as nn



class CFIM(nn.Module):
    def __init__(self, dim_text=768, dim_video=256, dim_audio=768, k_text=5, k_video=10, k_audio=20):
        super().__init__()
        # 投影层
        self.video2text = nn.Linear(dim_video, dim_text)
        self.audio2text = nn.Linear(dim_audio, dim_text)

        self.text2video = nn.Linear(dim_text, dim_video)
        self.audio2video = nn.Linear(dim_audio, dim_video)

        self.text2audio = nn.Linear(dim_text, dim_audio)
        self.video2audio = nn.Linear(dim_video, dim_audio)

        self.k_text, self.k_video, self.k_audio = k_text, k_video, k_audio

    def forward(self, X_text, X_video, X_audio):
        # Step 1: 显著性 (简单用L2 norm)
        score_text  = torch.norm(X_text, dim=-1)   # [30]
        score_video = torch.norm(X_video, dim=-1)  # [230]
        score_audio = torch.norm(X_audio, dim=-1)  # [480]

        # Step 2: top-K
        topk_text  = X_text[score_text.topk(self.k_text).indices]    # [K, 768]
        topk_video = X_video[score_video.topk(self.k_video).indices] # [K, 256]
        topk_audio = X_audio[score_audio.topk(self.k_audio).indices] # [K, 768]

        # Step 3: 投影
        vid2txt = self.video2text(topk_video)  # [K, 768]
        aud2txt = self.audio2text(topk_audio)  # [K, 768]

        txt2vid = self.text2video(topk_text)   # [K, 256]
        aud2vid = self.audio2video(topk_audio) # [K, 256]

        txt2aud = self.text2audio(topk_text)   # [K, 768]
        vid2aud = self.video2audio(topk_video) # [K, 768]

        # Step 4: Pad (简单做法 = 取平均加到序列上，实际可替换为稀疏位置注入)
        X_text_new  = X_text  + vid2txt.mean(0, keepdim=True)  + aud2txt.mean(0, keepdim=True)
        X_video_new = X_video + txt2vid.mean(0, keepdim=True)  + aud2vid.mean(0, keepdim=True)
        X_audio_new = X_audio + txt2aud.mean(0, keepdim=True)  + vid2aud.mean(0, keepdim=True)

        return X_text_new, X_video_new, X_audio_new
