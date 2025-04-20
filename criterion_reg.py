#---------------------------------------
#Since : 2024/09/05
#Update: 2024/12/16
# -*- coding: utf-8 -*-
#---------------------------------------
import torch
import torch.nn.functional as F

def evaluate_model(model, data_loader, device):
    """
    回帰タスクにおける評価関数
    指定されたデータローダー上で、モデルの出力に対して MSE, MAE, R² を計算し、
    全体の平均値を返します。

    Args:
        model (nn.Module): 評価対象のモデル
        data_loader (DataLoader): 評価用のデータローダー
        device (str): 使用するデバイス ("cpu" または "cuda")
        
    Returns:
        dict: 各評価指標の平均値が格納された辞書 (例: {'mse': 12.34, 'mae': 2.56, 'r2': 0.87})
    """
    # モデルを評価モードに設定（Dropout や BatchNorm の動作が変わるため）
    model.eval()
    
    # 各評価指標の累積値を初期化
    total_mse = 0.0  # 累積 MSE
    total_mae = 0.0  # 累積 MAE
    total_samples = 0  # 全サンプル数のカウント

    # R² の計算のために全バッチの予測値と正解値を蓄積するリストを用意
    all_preds = []
    all_targets = []

    # 評価時は勾配計算を無効にして高速化する
    with torch.no_grad():
        # データローダーからミニバッチごとにデータを取得
        for batch_x, batch_y in data_loader:
            # 入力と正解ラベルを指定デバイスに転送
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 入力データをフラット化してモデルに通す（必要に応じて変更）
            outputs = model(batch_x.flatten(1))
            
            # バッチサイズを取得して全体のサンプル数に加算
            B = batch_x.shape[0]
            total_samples += B

            # バッチごとに MSE と MAE を計算（reduction='mean' で平均を取得）
            mse_batch = F.mse_loss(outputs, batch_y, reduction='mean').item()
            mae_batch = F.l1_loss(outputs, batch_y, reduction='mean').item()
            
            # バッチサイズで重み付けして累積
            total_mse += mse_batch * B
            total_mae += mae_batch * B

            # R² の計算用に、バッチ内の予測値と正解値をリストに追加
            all_preds.append(outputs)
            all_targets.append(batch_y)

    # 全体の平均 MSE, MAE を計算
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    # 蓄積した各バッチの予測値と正解値を結合して全サンプル分のテンソルにする
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # R² の計算
    eps = 1e-8  # ゼロ除算防止のための微小値
    ss_res = torch.sum((all_targets - all_preds) ** 2)  # 残差平方和
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)  # 全変動平方和
    r2 = 1 - ss_res / (ss_tot + eps)  # 決定係数 R² の計算
    r2_value = r2.item()  # テンソルからスカラーに変換

    # 各評価指標の平均値を辞書として返す
    return avg_mse, avg_mae, r2_value
