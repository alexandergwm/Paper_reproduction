import torch
import torch.nn as nn
from typing import Optional

class ASDmodel(nn.Module):
    """
    自编码器模型，用于异常声音检测
    结构: 128*128*128*128*8*128*128*128*128
    """
    def __init__(self, input_dim: int):
        """
        初始化模型
        
        Args:
            input_dim (int): 输入特征维度
        """
        super(ASDmodel, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 8 (瓶颈层)
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 128
            nn.Linear(8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 输出层
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 重构输出，形状为 [batch_size, input_dim]
        """
        # 编码
        encoded = self.encoder(x)
        # 解码
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        只进行编码，用于特征提取
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 编码后的特征，形状为 [batch_size, 8]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        只进行解码，用于从编码特征重构
        
        Args:
            z (torch.Tensor): 编码特征，形状为 [batch_size, 8]
            
        Returns:
            torch.Tensor: 重构输出，形状为 [batch_size, input_dim]
        """
        return self.decoder(z)

def load_model(file_path: str, device: Optional[str] = None) -> ASDmodel:
    """
    加载模型
    
    Args:
        file_path (str): 模型文件路径
        device (str, optional): 设备类型 ('cuda' 或 'cpu')
        
    Returns:
        ASDmodel: 加载的模型
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model = torch.load(file_path, map_location=device)
    model.to(device)
    model.eval()
    return model

# 使用示例
if __name__ == "__main__":
    # 创建模型
    input_dim = 128 * 5  # 假设输入维度是 128 * 5
    model = ASDmodel(input_dim)
    
    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试编码和解码
    encoded = model.encode(x)
    decoded = model.decode(encoded)
    print(f"编码特征形状: {encoded.shape}")
    print(f"解码输出形状: {decoded.shape}")
    
    # 计算重构误差
    reconstruction_error = torch.mean((x - decoded) ** 2)
    print(f"重构误差: {reconstruction_error.item():.4f}") 