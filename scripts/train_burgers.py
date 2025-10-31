import sys
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU'}")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 获取项目根目录
# print(f"Project root directory: {project_root}")
# print(f"Current directory: {current_dir}")
sys.path.insert(0, project_root)
# Read the configuration
config_name = "default"
# Read the configuration
from zencfg import make_config_from_cli 
from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, WeightedSumLoss, Trainer, get_model
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.utils import get_wandb_api_key, count_model_params, get_project_root
from config.burgers_config import Default


config = make_config_from_cli(Default)
config = config.to_dict()

# print("config:", config)
# Set-up distributed communication, if using
device, is_logger = setup(config)
# Set up WandB logging
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.model.model_arch,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)
else: 
    wandb_init_args = None
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger



# Print config to screen
if config.verbose:
    print("##### CONFIG ######")
    print(config)
    sys.stdout.flush()

data_path = get_project_root() / config.data.folder
# Load the Burgers dataset
train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(data_path=data_path,
        n_train=config.data.n_train, batch_size=config.data.batch_size, 
        n_test=config.data.n_tests[0], test_batch_size=config.data.test_batch_sizes[0],
        temporal_subsample=config.data.get("temporal_subsample", 1),
        spatial_subsample=config.data.get("spatial_subsample", 1),
        )

model = get_model(config)
model.from_checkpoint(save_folder='./checkpoints', save_name='best_model')


# # Use distributed data parallel
# if config.distributed.use_distributed:
#     model = DDP(
#         model, device_ids=[device.index], output_device=device.index, static_graph=True
#     )

# # Create the optimizer
# optimizer = AdamW(
#     model.parameters(),
#     lr=config.opt.learning_rate,
#     weight_decay=config.opt.weight_decay,
# )

# if config.opt.scheduler == "ReduceLROnPlateau":
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         factor=config.opt.gamma,
#         patience=config.opt.scheduler_patience,
#         mode="min",
#     )
# elif config.opt.scheduler == "CosineAnnealingLR":
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=config.opt.scheduler_T_max
#     )
# elif config.opt.scheduler == "StepLR":
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
#     )
# else:
#     raise ValueError(f"Got scheduler={config.opt.scheduler}")


# # Creating the losses
# l2loss = LpLoss(d=2, p=2)
# h1loss = H1Loss(d=2)
# ic_loss = ICLoss()
# equation_loss = BurgersEqnLoss(method=config.opt.get('pino_method', 'fdm'), 
#                                visc=0.01, loss=F.mse_loss)

# training_loss = config.opt.training_loss
# if not isinstance(training_loss, (tuple, list)):
#     training_loss = [training_loss]

# losses = []
# weights = []
# for loss in training_loss:
#     # Append loss
#     if loss == 'l2':
#         losses.append(l2loss)
#     elif loss == 'h1':
#         losses.append(h1loss)
#     elif loss == 'equation':
#         losses.append(equation_loss)
#     elif loss == 'ic':
#         losses.append(ic_loss)
#     else:
#         raise ValueError(f'Training_loss={loss} is not supported.')

#     # Append loss weight
#     if "loss_weights" in config.opt:
#         weights.append(config.opt.loss_weights.get(loss, 1.))
#     else:
#         weights.append(1.)

# train_loss = WeightedSumLoss(losses=losses, weights=weights)
# eval_losses = {"h1": h1loss, "l2": l2loss}

# if config.verbose:
#     print("\n### MODEL ###\n", model)
#     print("\n### OPTIMIZER ###\n", optimizer)
#     print("\n### SCHEDULER ###\n", scheduler)
#     print("\n### LOSSES ###")
#     print(f"\n * Train: {train_loss}")
#     print(f"\n * Test: {eval_losses}")
#     print(f"\n### Beginning Training...\n")
#     sys.stdout.flush()

# # only perform MG patching if config patching levels > 0
# if config.patching.levels > 0:
#     data_processor = MGPatchingDataProcessor(model=model,
#                                         levels=config.patching.levels,
#                                         padding_fraction=config.patching.padding,
#                                         stitching=config.patching.stitching,
#                                         device=device,
#                                         in_normalizer=data_processor.in_normalizer,
#                                         out_normalizer=data_processor.out_normalizer)

# trainer = Trainer(
#     model=model,
#     n_epochs=config.opt.n_epochs,
#     data_processor=data_processor,
#     device=device,
#     mixed_precision=config.opt.mixed_precision,
#     eval_interval=config.opt.eval_interval,
#     log_output=config.wandb.log_output,
#     use_distributed=config.distributed.use_distributed,
#     verbose=config.verbose,
#     wandb_log = config.wandb.log
# )

# # Log parameter count
# if is_logger:
#     n_params = count_model_params(model)

#     if config.verbose:
#         print(f"\nn_params: {n_params}")
#         sys.stdout.flush()

#     if config.wandb.log:
#         to_log = {"n_params": n_params}
#         if config.n_params_baseline is not None:
#             to_log["n_params_baseline"] = (config.n_params_baseline,)
#             to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
#             to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
#         wandb.log(to_log, commit=False)
#         wandb.watch(model)


# trainer.train(
#     train_loader,
#     test_loaders,
#     optimizer,
#     scheduler,
#     regularizer=False,
#     training_loss=train_loss,
#     save_best='16_l2',  # 关键参数：监控的指标名称
#     eval_losses=eval_losses,
#     save_dir='./checkpoints'  # 保存路径
# )

# if config.wandb.log and is_logger:
#     wandb.finish()



# %%
# .. _plot_preds :
# Visualizing predictions
# ------------------------
# Let's take a look at what our model's predicted outputs look like. 
# Again note that in this example, we train on a very small resolution for
# a very small number of epochs.
# In practice, we would train at a larger resolution, on many more samples.

print("test_loader",  test_loaders.items())

# for loader_name, loader in test_loaders.items():
#     print(f"Evaluating on test loader: {loader_name}")
#     print("Number of samples:", len(loader))
#     with torch.no_grad():
#             for idx, sample in enumerate(loader):
#                 print(f"Sample {idx}:")
#                 sample = data_processor.preprocess(sample)
#                 print("After preprocessing:")
#                 print("x shape:", sample['x'].shape)    
#                 print("y shape:", sample['y'].shape)
#                 out = model(**sample)
#                 print("Output shape:", out.shape)
#                 loss = F.mse_loss(out, sample['y'])
#                 print("MSE Loss:", loss.item())

# test_samples = test_loaders[16].dataset
# print("test_samples", len(test_samples))
# # 调试：查看数据形状
# data = test_samples[0]
# print(f"Raw data shapes:")
# print(f"x shape: {data['x'].shape}")
# print(f"y shape: {data['y'].shape}")
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_test_predictions(model, test_loaders, data_processor, max_batches=2, samples_per_batch=2):
    """
    可视化测试预测结果
    """
    model.eval()
    
    print("test_loader items:", test_loaders.items())

    for loader_name, loader in test_loaders.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on test loader: {loader_name}")
        print(f"{'='*60}")
        
        all_samples = []
        all_predictions = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for idx, sample in enumerate(loader):
                print(f"\nSample {idx}:")
                sample = data_processor.preprocess(sample)
                print("After preprocessing:")
                print("x shape:", sample['x'].shape)    
                print("y shape:", sample['y'].shape)
                
                out = model(**sample)
                print("Output shape:", out.shape)
                
                loss = F.mse_loss(out, sample['y'])
                print("MSE Loss:", loss.item())
                
                # 保存数据用于可视化
                all_samples.append(sample)
                all_predictions.append(out.cpu())
                all_targets.append(sample['y'].cpu())
                all_losses.append(loss.item())
                
                # 限制批次数量
                if idx >= max_batches - 1:
                    break
        
        # 可视化这个loader的结果
        visualize_loader_results(loader_name, all_samples, all_predictions, all_targets, all_losses, samples_per_batch)

def visualize_loader_results(loader_name, samples, predictions, targets, losses, samples_per_batch=2):
    """
    可视化单个loader的结果
    """
    print(f"\nVisualizing results for {loader_name}...")
    
    num_batches = len(samples)
    total_samples_to_show = num_batches * samples_per_batch
    
    # 创建大图 - 每行显示一个样本
    fig, axes = plt.subplots(total_samples_to_show, 4, figsize=(20, 5 * total_samples_to_show))
    
    # 如果只有一行，调整axes维度
    if total_samples_to_show == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    for batch_idx in range(num_batches):
        sample = samples[batch_idx]
        pred = predictions[batch_idx]
        target = targets[batch_idx]
        loss = losses[batch_idx]
        
        batch_size = sample['x'].shape[0]
        
        # 每个批次显示前samples_per_batch个样本
        num_show = min(samples_per_batch, batch_size)
        
        for sample_idx in range(num_show):
            # 获取单个样本数据
            x_single = sample['x'][sample_idx].cpu().squeeze()
            y_true_single = target[sample_idx].squeeze()
            y_pred_single = pred[sample_idx].squeeze()
            
            print(f"Batch {batch_idx}, Sample {sample_idx} shapes - x: {x_single.shape}, y_true: {y_true_single.shape}, y_pred: {y_pred_single.shape}")
            
            # 绘制输入 x
            ax1 = axes[plot_idx, 0] if total_samples_to_show > 1 else axes[0]
            if len(x_single.shape) == 2:
                im1 = ax1.imshow(x_single.numpy(), cmap='viridis', aspect='auto')
                ax1.set_title(f'Batch {batch_idx} Sample {sample_idx}\nInput x\nShape: {x_single.shape}')
                plt.colorbar(im1, ax=ax1)
            else:
                # 处理1D数据
                ax1.plot(x_single.numpy(), 'b-', linewidth=2)
                ax1.set_title(f'Batch {batch_idx} Sample {sample_idx}\nInput x\nShape: {x_single.shape}')
            ax1.grid(True, alpha=0.3)
            
            # 绘制真实值 y
            ax2 = axes[plot_idx, 1] if total_samples_to_show > 1 else axes[1]
            if len(y_true_single.shape) == 2:
                im2 = ax2.imshow(y_true_single.numpy(), cmap='viridis', aspect='auto')
                ax2.set_title(f'Ground Truth\nShape: {y_true_single.shape}')
                plt.colorbar(im2, ax=ax2)
            else:
                ax2.plot(y_true_single.numpy(), 'g-', linewidth=2, label='Ground Truth')
                ax2.set_title(f'Ground Truth\nShape: {y_true_single.shape}')
            ax2.grid(True, alpha=0.3)
            
            # 绘制预测值
            ax3 = axes[plot_idx, 2] if total_samples_to_show > 1 else axes[2]
            if len(y_pred_single.shape) == 2:
                im3 = ax3.imshow(y_pred_single.numpy(), cmap='viridis', aspect='auto')
                ax3.set_title(f'Prediction\nShape: {y_pred_single.shape}')
                plt.colorbar(im3, ax=ax3)
            else:
                ax3.plot(y_pred_single.numpy(), 'r-', linewidth=2, label='Prediction')
                ax3.set_title(f'Prediction\nShape: {y_pred_single.shape}')
            ax3.grid(True, alpha=0.3)
            
            # 绘制误差
            ax4 = axes[plot_idx, 3] if total_samples_to_show > 1 else axes[3]
            error = np.abs(y_true_single.numpy() - y_pred_single.numpy())
            if len(error.shape) == 2:
                im4 = ax4.imshow(error, cmap='hot', aspect='auto')
                ax4.set_title(f'Absolute Error\nBatch MSE: {loss:.6f}')
                plt.colorbar(im4, ax=ax4)
            else:
                ax4.plot(error, 'orange', linewidth=2, label='Error')
                ax4.set_title(f'Absolute Error\nBatch MSE: {loss:.6f}')
                ax4.fill_between(range(len(error)), error, alpha=0.3, color='orange')
            ax4.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.suptitle(f'Test Results for {loader_name}\n'
                f'Total Batches: {num_batches}, Average MSE: {np.mean(losses):.6f}', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_single_batch_comparison(model, test_loaders, data_processor, batch_idx=0):
    """
    可视化单个批次的详细比较
    """
    model.eval()
    
    for loader_name, loader in test_loaders.items():
        print(f"\nVisualizing single batch for {loader_name}...")
        
        with torch.no_grad():
            for idx, sample in enumerate(loader):
                if idx == batch_idx:
                    sample = data_processor.preprocess(sample)
                    out = model(**sample)
                    loss = F.mse_loss(out, sample['y'])
                    
                    print(f"Batch {idx} - x: {sample['x'].shape}, y: {sample['y'].shape}, out: {out.shape}")
                    
                    # 可视化这个批次的所有样本
                    visualize_batch_details(loader_name, sample, out, loss, batch_idx)
                    break

def visualize_batch_details(loader_name, sample, predictions, batch_loss, batch_idx):
    """
    可视化单个批次的详细信息
    """
    batch_size = sample['x'].shape[0]
    
    # 创建大图：每行一个样本，每列一个时间步的对比
    fig, axes = plt.subplots(batch_size, 3, figsize=(18, 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(batch_size):
        x_single = sample['x'][sample_idx].cpu().squeeze()  # [17, 16]
        y_true_single = sample['y'][sample_idx].cpu().squeeze()  # [17, 16]
        y_pred_single = predictions[sample_idx].cpu().squeeze()  # [17, 16]
        
        # 选择几个时间步进行对比
        time_steps = [0, y_true_single.shape[0]//2, -1]  # 开始、中间、结束
        
        for col, t in enumerate(time_steps):
            ax = axes[sample_idx, col] if batch_size > 1 else axes[col]
            
            # 获取特定时间步的数据
            if len(y_true_single.shape) == 2:
                y_true_t = y_true_single[t].numpy()
                y_pred_t = y_pred_single[t].numpy()
            else:
                y_true_t = y_true_single.numpy()
                y_pred_t = y_pred_single.numpy()
            
            # 绘制对比
            ax.plot(y_true_t, 'g-', linewidth=2, label='Ground Truth')
            ax.plot(y_pred_t, 'r--', linewidth=2, label='Prediction')
            ax.fill_between(range(len(y_true_t)), y_true_t, y_pred_t, alpha=0.3, color='gray')
            ax.set_title(f'Sample {sample_idx}, Time {t}\nMSE: {F.mse_loss(torch.tensor(y_pred_t), torch.tensor(y_true_t)):.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if sample_idx == batch_size - 1:
                ax.set_xlabel('Spatial Position')
            if col == 0:
                ax.set_ylabel('Value')
    
    plt.suptitle(f'{loader_name} - Batch {batch_idx} Detailed Comparison\nBatch MSE: {batch_loss:.6f}', 
                fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def create_error_analysis(model, test_loaders, data_processor):
    """
    创建误差分析图
    """
    model.eval()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    all_errors = []
    all_predictions = []
    all_targets = []
    loader_names = []
    
    for loader_name, loader in test_loaders.items():
        print(f"Analyzing errors for {loader_name}...")
        
        loader_errors = []
        loader_predictions = []
        loader_targets = []
        
        with torch.no_grad():
            for idx, sample in enumerate(loader):
                sample = data_processor.preprocess(sample)
                out = model(**sample)
                
                # 计算每个样本的误差
                batch_errors = torch.abs(out - sample['y']).flatten().cpu().numpy()
                loader_errors.extend(batch_errors)
                
                loader_predictions.extend(out.cpu().flatten().numpy())
                loader_targets.extend(sample['y'].cpu().flatten().numpy())
                
                if idx >= 1:  # 限制数据量
                    break
        
        all_errors.append(loader_errors)
        all_predictions.append(loader_predictions)
        all_targets.append(loader_targets)
        loader_names.append(loader_name)
    
    # 1. 误差分布箱线图
    axes[0, 0].boxplot(all_errors, labels=loader_names)
    axes[0, 0].set_title('Error Distribution by Loader')
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 预测vs真实值散点图
    for i, (preds, targets) in enumerate(zip(all_predictions, all_targets)):
        axes[0, 1].scatter(targets[:1000], preds[:1000], alpha=0.5, s=10, label=loader_names[i])
    
    min_val = min(min(all_targets[0]), min(all_predictions[0]))
    max_val = max(max(all_targets[0]), max(all_predictions[0]))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Ground Truth')
    axes[0, 1].set_ylabel('Predictions')
    axes[0, 1].set_title('Predictions vs Ground Truth')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 误差直方图
    for i, errors in enumerate(all_errors):
        axes[1, 0].hist(errors, bins=50, alpha=0.6, label=loader_names[i])
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Histogram')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累积误差分布
    for i, errors in enumerate(all_errors):
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cdf, label=loader_names[i], linewidth=2)
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Error Analysis Summary', fontsize=16, y=0.98)
    plt.show()

# 使用示例
# 方法1: 基本可视化（限制批次和样本数量）
visualize_test_predictions(model, test_loaders, data_processor, max_batches=2, samples_per_batch=2)

# 方法2: 单个批次详细比较
visualize_single_batch_comparison(model, test_loaders, data_processor, batch_idx=0)

# 方法3: 误差分析
create_error_analysis(model, test_loaders, data_processor)