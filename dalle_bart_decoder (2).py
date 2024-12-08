#KEEP TITLE SAME TO ENSURE COMPATIBILITY

from diffusers import AutoencoderKL

class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        device: str,
    ):
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.image_vocab_count = image_vocab_count
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embed_count)
        self.layers: List[DecoderLayer] = nn.ModuleList(
            [
                DecoderLayer(
                    head_count=attention_head_count,
                    embed_count=embed_count,
                    glu_embed_count=glu_embed_count,
                    device=device,
                )
                for _ in range(layer_count)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)

#Initialize VAE
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    def forward(
        self,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
#Accept VAE Latents
        prev_latents: FloatTensor,
        token_index: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index = token_index.unsqueeze(0).repeat(image_count * 2, 1)

        decoder_state = self.embed_positions(token_index)
        decoder_state += prev_latents
        decoder_state = self.layernorm_embedding(decoder_state)

        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index,
            )

        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        return logits, attention_state

    def sample_tokens(self, settings, **kwargs) -> Tuple[FloatTensor, FloatTensor]:
        logits, attention_state = self.forward(**kwargs)
        image_count = logits.shape[0] // 2
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]

        logits = logits[:, -1, :]
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        logits -= logits_sorted[:, [0]]
        logits /= temperature
        logits.exp_()
        logits *= is_kept.to(torch.float32)

#Use VAE latent sampling
        latents = torch.multinomial(logits, 1)[:, 0]
        return latents, attention_state

    def decode_latents(self, latents: FloatTensor) -> torch.Tensor:
        """Decode latents back into images using VAE."""
        with torch.no_grad():
            decoded_images = self.vae.decode(latents).sample
        return decoded_images
