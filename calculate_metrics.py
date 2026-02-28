import pandas as pd
import numpy as np
import glob
import os
import warnings
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, accuracy_score
from sklearn.utils import resample

# 忽略 y_pred 包含 y_true 中没有的类别的警告
warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')

def bootstrap_metrics(y_true, y_pred, n_iterations=1000, alpha=0.95):
    """
    使用 Bootstrap 方法计算 Macro-F1, Balanced Accuracy 和 Accuracy 的置信区间和标准差
    """
    n_size = len(y_true)
    f1_scores = []
    bacc_scores = []
    acc_scores = []
    
    # 确保 y_true 和 y_pred 是 numpy 数组，方便索引
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    for _ in range(n_iterations):
        # 有放回重采样
        indices = resample(np.arange(n_size), n_samples=n_size)
        y_true_boot = y_true_arr[indices]
        y_pred_boot = y_pred_arr[indices]
        
        # 计算指标
        f1 = f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        bacc = balanced_accuracy_score(y_true_boot, y_pred_boot)
        acc = accuracy_score(y_true_boot, y_pred_boot)
        
        f1_scores.append(f1)
        bacc_scores.append(bacc)
        acc_scores.append(acc)
        
    # 计算标准差
    f1_std = np.std(f1_scores)
    bacc_std = np.std(bacc_scores)
    acc_std = np.std(acc_scores)
    
    # 计算置信区间
    p = ((1.0 - alpha) / 2.0) * 100
    lower_p = p
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    f1_ci = (np.percentile(f1_scores, lower_p), np.percentile(f1_scores, upper_p))
    bacc_ci = (np.percentile(bacc_scores, lower_p), np.percentile(bacc_scores, upper_p))
    acc_ci = (np.percentile(acc_scores, lower_p), np.percentile(acc_scores, upper_p))
    
    return f1_std, f1_ci, bacc_std, bacc_ci, acc_std, acc_ci

def calculate_metrics():
    # 查找当前目录下所有的 csv 文件
    csv_files = glob.glob('*.csv')

    if not csv_files:
        print("当前目录下未找到CSV文件。")
        return

    # 存储所有模型的汇总结果
    all_model_summary = []

    for file_path in csv_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"正在处理模型: {model_name} ...")
        
        try:
            df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            if 'question_type' not in df.columns or 'gt_answer' not in df.columns or 'pred_option' not in df.columns:
                print(f"跳过 {file_path}: 缺少必要的列 ('question_type', 'gt_answer', 'pred_option')")
                continue
            
            # 过滤掉没有预测结果的行（如果有的话）
            df = df.dropna(subset=['gt_answer', 'pred_option'])
            
            # 计算整体的随机猜测下界 (按各属性样本量加权平均)
            total_support = len(df)
            expected_random_acc_sum = 0
            for q_type, group in df.groupby('question_type'):
                num_classes = group['gt_answer'].nunique()
                if num_classes > 0:
                    expected_random_acc_sum += len(group) * (1.0 / num_classes)
            overall_random_bound = expected_random_acc_sum / total_support if total_support > 0 else 0
            
            # 1. 整体 (Overall) 的 Macro-F1, Balanced Accuracy 和 Accuracy
            overall_macro_f1 = f1_score(df['gt_answer'], df['pred_option'], average='macro', zero_division=0)
            overall_balanced_acc = balanced_accuracy_score(df['gt_answer'], df['pred_option'])
            overall_acc = accuracy_score(df['gt_answer'], df['pred_option'])
            
            # 计算整体的 Bootstrap 置信区间和标准差
            overall_f1_std, overall_f1_ci, overall_bacc_std, overall_bacc_ci, overall_acc_std, overall_acc_ci = bootstrap_metrics(df['gt_answer'], df['pred_option'])
            
            all_model_summary.append({
                'Model': model_name,
                'Attribute': 'Overall (整体)',
                'Category': 'All',
                'Num Classes': 'Mixed',
                'Random Guess Bound': overall_random_bound,
                'Accuracy': overall_acc,
                'Accuracy Std': overall_acc_std,
                'Accuracy 95% CI': f"[{overall_acc_ci[0]:.4f}, {overall_acc_ci[1]:.4f}]",
                'Macro-F1 / F1': overall_macro_f1,
                'Macro-F1 Std': overall_f1_std,
                'Macro-F1 95% CI': f"[{overall_f1_ci[0]:.4f}, {overall_f1_ci[1]:.4f}]",
                'Balanced Acc / Recall': overall_balanced_acc,
                'Balanced Acc Std': overall_bacc_std,
                'Balanced Acc 95% CI': f"[{overall_bacc_ci[0]:.4f}, {overall_bacc_ci[1]:.4f}]",
                'Support': total_support
            })
            
            # 2. 按属性 (question_type) 计算
            for q_type, group in df.groupby('question_type'):
                num_classes = group['gt_answer'].nunique()
                random_bound = 1.0 / num_classes if num_classes > 0 else 0
                
                # 该属性下的 Macro-F1, Balanced Accuracy 和 Accuracy
                attr_macro_f1 = f1_score(group['gt_answer'], group['pred_option'], average='macro', zero_division=0)
                attr_balanced_acc = balanced_accuracy_score(group['gt_answer'], group['pred_option'])
                attr_acc = accuracy_score(group['gt_answer'], group['pred_option'])
                
                # 计算属性的 Bootstrap 置信区间和标准差
                attr_f1_std, attr_f1_ci, attr_bacc_std, attr_bacc_ci, attr_acc_std, attr_acc_ci = bootstrap_metrics(group['gt_answer'], group['pred_option'])
                
                all_model_summary.append({
                    'Model': model_name,
                    'Attribute': q_type,
                    'Category': 'All (Macro)',
                    'Num Classes': num_classes,
                    'Random Guess Bound': random_bound,
                    'Accuracy': attr_acc,
                    'Accuracy Std': attr_acc_std,
                    'Accuracy 95% CI': f"[{attr_acc_ci[0]:.4f}, {attr_acc_ci[1]:.4f}]",
                    'Macro-F1 / F1': attr_macro_f1,
                    'Macro-F1 Std': attr_f1_std,
                    'Macro-F1 95% CI': f"[{attr_f1_ci[0]:.4f}, {attr_f1_ci[1]:.4f}]",
                    'Balanced Acc / Recall': attr_balanced_acc,
                    'Balanced Acc Std': attr_bacc_std,
                    'Balanced Acc 95% CI': f"[{attr_bacc_ci[0]:.4f}, {attr_bacc_ci[1]:.4f}]",
                    'Support': len(group)
                })
                
                # 3. 按类别 (具体选项 A, B, C...) 计算 F1 和 Recall
                # classification_report 可以方便地获取每个类别的指标
                report = classification_report(group['gt_answer'], group['pred_option'], output_dict=True, zero_division=0)
                
                for label, metrics in report.items():
                    # 过滤掉 'accuracy', 'macro avg', 'weighted avg' 等非类别键
                    if label not in ['accuracy', 'macro avg', 'weighted avg']:
                        all_model_summary.append({
                            'Model': model_name,
                            'Attribute': q_type,
                            'Category': label,
                            'Num Classes': '-',
                            'Random Guess Bound': random_bound,
                            'Accuracy': '-',
                            'Accuracy Std': '-',
                            'Accuracy 95% CI': '-',
                            'Macro-F1 / F1': metrics['f1-score'],
                            'Macro-F1 Std': '-',
                            'Macro-F1 95% CI': '-',
                            'Balanced Acc / Recall': metrics['recall'], # 对于单个类别，Balanced Acc 相当于 Recall (Sensitivity)
                            'Balanced Acc Std': '-',
                            'Balanced Acc 95% CI': '-',
                            'Support': metrics['support']
                        })
                        
        except Exception as e:
            print(f"处理 {file_path} 时出错: {e}")

    if all_model_summary:
        # 转换为 DataFrame
        summary_df = pd.DataFrame(all_model_summary)
        
        # 保存为 CSV 文件
        output_file = 'metrics_summary_report.csv'
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n所有模型的指标计算完成！结果已保存至: {output_file}")
        
        # 打印部分结果预览
        print("\n--- 整体指标预览 ---")
        overall_df = summary_df[summary_df['Attribute'] == 'Overall (整体)']
        print(overall_df[['Model', 'Accuracy', 'Macro-F1 / F1', 'Balanced Acc / Recall']].to_string(index=False))

if __name__ == "__main__":
    calculate_metrics()
