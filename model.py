import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        import os
        
        # 절대 경로로 model 폴더 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_folder_path = os.path.join(current_dir, 'model')
        
        try:
            # 폴더가 없으면 생성 
            os.makedirs(model_folder_path, exist_ok=True)
            print(f"Model directory ensured: {model_folder_path}")
                
            file_path = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_path)
            print(f"Model saved successfully to: {file_path}")
            
            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"File size: {file_size} bytes")
                return True
            else:
                print("Warning: File was not created!")
                return False
                
        except PermissionError as e:
            print(f"Permission error: {e}")
            # 현재 디렉토리에 저장 시도
            try:
                fallback_path = os.path.join(current_dir, file_name)
                torch.save(self.state_dict(), fallback_path)
                print(f"Model saved to fallback location: {fallback_path}")
                return True
            except Exception as fallback_error:
                print(f"Fallback save failed: {fallback_error}")
                return False
        except Exception as e:
            print(f"Error saving model: {e}")
            return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
