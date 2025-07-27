import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_factor = 1/math.sqrt(self.head_dim) 

        self.to_qkv = nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=False)#保持對稱性和簡化計算，去掉bias:b
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, nums_token, token_dim = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size, nums_token, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        
        output = self._scaled_dot_product_attention(query, key, value)
        
        output = output.transpose(1, 2).reshape(batch_size, nums_token, token_dim)
        output = self.proj(output)
        return output
    
    def _scaled_dot_product_attention(self, q, k, v):
        """
        Establish the scaled dot-product attention mechanism.

        Args:
            q: query tensor of shape (batch_size, num_heads, num_query_tokens, d_k)
            k: key tensor of shape (batch_size, num_heads, num_key_tokens, d_k)
            v: value tensor of shape (batch_size, num_heads, num_value_tokens, d_v)
            
        Returns:
            Tensor of shape (batch_size, num_heads, num_query_tokens, d_v)
            The weighted sum of values after applying attention.
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale_factor
        attn = self.attn_drop(attn.softmax(dim=-1))
        return attn @ v

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    