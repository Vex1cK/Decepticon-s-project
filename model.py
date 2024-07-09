import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def scaled_softmax_attention(query, key, value):
    """
    Args:
        query: torch.Tensor (..., L, D)
        key: torch.Tensor (..., L, D)
        value: torch.Tensor (..., L, D)
    Returns:
        res: torch.Tensor (..., L, D), output of the attention layer (\softmax(Q K^T / d) V
        attention: torch.Tensor (..., L, L), attention weights (\softmax(Q K^T / d))

    L is the length of sequence, D is the embedding dimension
    """

    attention = F.softmax((query @ key.transpose(-2, -1) / math.sqrt(key.size(dim=1))), dim=-1)
    res = attention @ value

    return res, attention



class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Финальный проекционный слой
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
            return_attention: If specified, returns attention along with outputs
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """
        batch_size = x.size(dim=0)

        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        outputs, attention = scaled_softmax_attention(q, k, v)
        # Конкатенация всех голов и применение финального линейного слоя
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        outputs = self.o_proj(outputs)
        if return_attention:
            return outputs, attention
        else:
            return outputs
        

class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, feedforward_dim, activation=nn.ReLU(), dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.activation = activation
        self.dropout = dropout
        self.FeedForward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.multihead_attention = MultiheadAttention(embed_dim, num_heads)


    def forward(self, x, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)
        """

        var_out, var_att = self.multihead_attention(F.layer_norm(x, x.shape[1:]), return_attention=True)
        if self.dropout:
            var_out = F.dropout(var_out, self.dropout)
        var_out += x
        var_out = F.layer_norm(var_out, var_out.shape[1:])
        var_out = self.FeedForward(var_out)
        var_out += x



        outputs, attention = var_out, var_att

        if return_attention:
            return outputs, attention
        else:
            return outputs

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # Создаем матрицу позиционных кодировок заранее
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the module's state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
        Inputs
            x - Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Добавляем позиционные кодировки к каждому эмбеддингу
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerForSequenceClassification(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        activation = nn.GELU,
        max_len: int = 5000,
        dropout: float = 0.0
    ):
        super().__init__()
        # define layers
        self.cls_token = torch.randn(embed_dim) # TODO create vector of size (embed_dim,) from N(0, 1)
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.multh_att = MultiheadAttention(embed_dim, num_heads)
        encoder_blocks = [EncoderBlock(embed_dim, num_heads, feedforward_dim, activation(), dropout) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*encoder_blocks,)

        self.classifier = nn.Linear(embed_dim, num_classes,)

    def forward_attention(self, x):
        """
        This method returns attention maps from all encoder blocks

        Args:
            x: torch.Tensor (B, L, |V|)
        Returns:
            attn: List[torch.Tensor] (B, num_heads, L+1, L+1) x num_layers
        """
        attn_maps = []
        for layer in self.encoder.layers:
            _, attn = self.multh_att(x, return_attention = True)
            attn_maps.append(attn)
        return attn_maps

    def forward(self, x):
        """
        Args:
            x: torch.Tensor (B, L, |V|)
        Returns:
            x: torch.Tensor (B, |C|)
        """


        embed_x = self.input_embedding(x)
        pos_x = self.positional_encoding(embed_x)
        cls_tokens = self.cls_token.repeat(pos_x.size(0), 1, 1).to(device)
        pos_x = torch.cat((cls_tokens, pos_x), dim=1)

        encode_x = self.encoder(pos_x)


        cls_vector = encode_x[:, -1, :]
        x = self.classifier(cls_vector)

        return x

def get_model():
    model = TransformerForSequenceClassification(
        num_classes=1,
        input_dim=619,
        embed_dim=1024,
        num_heads=32,
        feedforward_dim=2024,
        activation=nn.ReLU,
        num_layers=20
    )
    model.load_state_dict(torch.load("model_1.pth", map_location=torch.device('cpu')))
    model = model.eval()
    return model