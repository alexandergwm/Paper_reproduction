import torch
import torch.nn as nn

class AutoEncoderLoss(nn.Module):
    """
    自编码器的MSE损失函数
    用于计算输入和重构输出之间的均方误差
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        计算重构损失
        
        Args:
            x (torch.Tensor): 原始输入
            x_recon (torch.Tensor): 重构输出
            
        Returns:
            torch.Tensor: MSE损失值
        """
        return self.mse_loss(x_recon, x)

# 使用示例
if __name__ == "__main__":
    # 创建损失函数
    loss_fn = AutoEncoderLoss()
    
    # 测试损失计算
    batch_size = 32
    input_dim = 640
    
    # 模拟输入和重构输出
    x = torch.randn(batch_size, input_dim)
    x_recon = torch.randn(batch_size, input_dim)
    
    # 计算损失
    loss = loss_fn(x, x_recon)
    print(f"MSE损失: {loss.item():.4f}")