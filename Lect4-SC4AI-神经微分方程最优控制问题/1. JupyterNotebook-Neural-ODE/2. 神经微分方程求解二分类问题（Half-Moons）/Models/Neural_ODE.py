import torch
import torch.nn as nn
import torchdiffeq
import pytorch_lightning as pl

# evaluation of right-hand-side function    
class _NODE_RHS(nn.Module):
    def __init__(self, module, autonomous=True):
        super().__init__()
        self.module = module
        self.autonomous = autonomous

    def forward(self, t, x):
        if not self.autonomous:
            x = torch.cat([torch.ones_like(x[:, [0]]) * t, x], 1)
        return self.module(x)


# define neural ODE model
class NODE(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = 'dopri5',
                 rtol: float = 1e-4, atol: float = 1e-4, adjoint: bool = True,
                 autonomous: bool = True):
        super().__init__()
        self.odefunc = _NODE_RHS(odefunc, autonomous=autonomous)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        self.integration_time = torch.tensor([0, 1], dtype=torch.float32)  
    
    @property
    def ode_method(self):
        return torchdiffeq.odeint_adjoint if self.use_adjoint else torchdiffeq.odeint

    def forward(self, x: torch.Tensor, adjoint: bool = True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        integration_time = integration_time.to(x.device)
        ode_method =  torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        out = ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver)
        return out

# train/test neural ODE model
class NODE_Trainer(pl.LightningModule):
    def __init__(self, model:nn.Module, t_span:torch.Tensor, learning_rate:float=5e-3):
        super().__init__()
        self.model = model
        self.t_span = t_span
        self.learning_rate = learning_rate
        self.train_loss = []
        self.test_loss = []
    
    def forward(self, x):
        return self.model(x)

    def inference(self, x, time_span):
        x_output = self.model(x, adjoint=False, integration_time=time_span)
        y_output = torch.nn.functional.softmax(x_output, dim=-1)
        return x_output, y_output
   
    def training_step(self, batch, batch_idx):
        x, y = batch      
        y_pred = self(x)
        y_pred = y_pred[-1]  # select terminal point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_pred, y) 
        self.train_loss.append(loss.item())       
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch      
        y_pred = self(x)
        y_pred = y_pred[-1]  # select last point of solution trajectory
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.test_loss.append(loss.item())  
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer        