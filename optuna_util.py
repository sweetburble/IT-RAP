import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import datetime
import shutil
import json
from sklearn.cluster import KMeans
import time
import gc
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_perturbation(perturbation):

    perturbation_no_batch = perturbation.squeeze(0)


    squared_perturbation = perturbation_no_batch ** 2
    sum_squared_perturbation = torch.sum(squared_perturbation, dim=0)
    euclidean_norm_perturbation = torch.sqrt(sum_squared_perturbation)


    perturbation_array = euclidean_norm_perturbation.cpu().numpy()

    return perturbation_array

"""
Prints various debug logs.
x_real, perturbed_image, original_gen_image, perturbed_gen_image: shape [1, 3, 256, 256] / values [-1, 1]
"""
def print_debug(x_real, perturbed_image, original_gen_image, perturbed_gen_image):

    print(f'x_real shape: {x_real.shape}, perturbed_image shape: {perturbed_image.shape}, perturbed_image type: {type(perturbed_image)}')


    real_red_min = x_real[0, 0].min()
    real_red_max = x_real[0, 0].max()
    real_green_min = x_real[0, 1].min()
    real_green_max = x_real[0, 1].max()
    real_blue_min = x_real[0, 2].min()
    real_blue_max = x_real[0, 2].max()
    print(f"Original Image Red Min: {real_red_min}, Max: {real_red_max}")
    print(f"Original Image Green Min: {real_green_min}, Max: {real_green_max}")
    print(f"Original Image Blue Min: {real_blue_min}, Max: {real_blue_max}")


    adv_red_min = perturbed_image[0, 0].min()
    adv_red_max = perturbed_image[0, 0].max()
    adv_green_min = perturbed_image[0, 1].min()
    adv_green_max = perturbed_image[0, 1].max()
    adv_blue_min = perturbed_image[0, 2].min()
    adv_blue_max = perturbed_image[0, 2].max()
    print(f"Adversarial Example Red Min: {adv_red_min}, Max: {adv_red_max}")
    print(f"Adversarial Example Green Min: {adv_green_min}, Max: {adv_green_max}")
    print(f"Adversarial Example Blue Min: {adv_blue_min}, Max: {adv_blue_max}")


def calculate_and_save_metrics(
    original_gen_image,
    perturbed_gen_image,
    transform_type,
    results,
    success_l2_threshold: float = 0.05,
):

    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)


    l1_error = F.l1_loss(perturbed_gen_image, original_gen_image)
    l2_error = F.mse_loss(perturbed_gen_image, original_gen_image)

    results[transform_type]["l1_error"] += l1_error.item()
    results[transform_type]["l2_error"] += l2_error.item()


    if l2_error > success_l2_threshold:
        results[transform_type]["attack_success"] += 1


    original_gen_image_np = original_gen_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    perturbed_gen_image_np = perturbed_gen_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    defense_lpips = lpips_loss(original_gen_image, perturbed_gen_image).mean()
    defense_psnr = psnr(original_gen_image_np, perturbed_gen_image_np, data_range=2.0)
    defense_ssim = ssim(original_gen_image_np, perturbed_gen_image_np, data_range=2.0, win_size=3, channel_axis=2)

    results[transform_type]["defense_lpips"] += defense_lpips.item()
    results[transform_type]["defense_psnr"] += defense_psnr
    results[transform_type]["defense_ssim"] += defense_ssim

    return results

"""
Prints the final result metrics.
x_real, perturbed_image, original_gen_image, perturbed_gen_image: shape [1, 3, 256, 256] / values [-1, 1]
"""
def print_final_metrics(episode, total_perturbation_map, total_remain_map, total_l1_error, total_l2_error, attack_success, invisible_psnr, invisible_ssim, invisible_lpips, defense_psnr, defense_ssim, defense_lpips):
    print("\n======== Printing Metrics ========")

    average_perturbation_map = total_perturbation_map / episode

    print(f"--- Average Perturbation Over {episode} Iterations ---")

    print("Average Perturbation Map (256x256 array):\n", average_perturbation_map)


    total_average_perturbation_value = np.sum(average_perturbation_map)
    print("\nTotal Average Perturbation Value (Sum of all pixels in average map) : ", total_average_perturbation_value)


    average_remain_map = total_remain_map / episode
    print("\nAverage Remain Perturbation Map (256x256 array):\n", average_remain_map)

    total_average_remain_perturbation_value = np.sum(average_remain_map)
    print("\nTotal Average Remain Perturbation Value (Sum of all pixels in remain map) : ", total_average_remain_perturbation_value)


    print(f'{episode} images. L1 error: {total_l1_error / episode:.5f}. L2 error: {total_l2_error / episode:.5f}. prop_dist: {float(attack_success) / episode:.5f}.')

    print(f'Invisibility PSNR: {invisible_psnr / episode:.5f} dB. Invisibility SSIM: {invisible_ssim / episode:.5f}. Invisibility LPIPS: {invisible_lpips / episode:.5f}\nDeepfake Defense PSNR: {defense_psnr / episode:.5f} dB. Deepfake Defense SSIM: {defense_ssim / episode:.5f}. Deepfake Defense LPIPS: {defense_lpips / episode:.5f}')


"""
Print and save final metrics for all transformation methods.
"""

def print_comprehensive_metrics(results, episode, total_invisible_psnr, total_invisible_ssim, total_invisible_lpips, combo_index=None):

    output_lines = []


    header1 = "\n" + "="*100
    header2 = "Final Results Summary by Image Transformation Method"
    header3 = "="*100

    print(header1)
    print(header2)
    print(header3)
    output_lines.extend([header1, header2, header3])


    if combo_index is not None:
        combo_line = f"Hyperparameter combo index: {combo_index}"
        print(combo_line)
        output_lines.append(combo_line)


    invisibility_line = f'Invisibility PSNR: {total_invisible_psnr / episode:.5f} dB. Invisibility SSIM: {total_invisible_ssim / episode:.5f}. Invisibility LPIPS: {total_invisible_lpips / episode:.5f}'
    print(invisibility_line)
    output_lines.append(invisibility_line)

    separator = "="*100
    print(separator)
    output_lines.append(separator)


    table_header = f"{'Transform Type':<15} | {'L1 Error':<10} | {'L2 Error':<10} | {'Noise Attack PSNR':<18} | {'Noise Attack SSIM':<18} | {'Noise Attack LPIPS':<18} | {'Success Rate (%)':<18} | {'Noise Residue'}"
    table_separator = "-"*130

    print(table_header)
    print(table_separator)
    output_lines.append(table_header)
    output_lines.append(table_separator)

    total_success_rate = 0.0

    for transform_type, metrics in results.items():

        avg_l1 = metrics["l1_error"] / episode
        avg_l2 = metrics["l2_error"] / episode
        avg_def_psnr = metrics["defense_psnr"] / episode
        avg_def_ssim = metrics["defense_ssim"] / episode
        avg_def_lpips = metrics["defense_lpips"] / episode
        success_rate = metrics["attack_success"] / episode

        total_success_rate += success_rate

        average_remain_map = metrics["total_remain_map"] / episode
        average_remain_perturbation_value = np.sum(average_remain_map)


        result_line = (f"{transform_type:<15} | {avg_l1:<10.4f} | {avg_l2:<10.4f} | "
                       f"{avg_def_psnr:<18.2f} | {avg_def_ssim:<18.4f} | {avg_def_lpips:<18.4f} | "
                       f"{success_rate:<18.2f} | {average_remain_perturbation_value:.2f}")
        print(result_line)
        output_lines.append(result_line)

    print(separator)
    output_lines.append(separator)


    output_dir = "result_test"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"training_performance.txt")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\n[Success] Results saved to file: {file_path}")
    except IOError as e:
        print(f"\n[Error] Failed to save file: {e}")

    print("="*100)


    avg_invisible_psnr = total_invisible_psnr / episode
    avg_invisible_ssim = total_invisible_ssim / episode
    avg_invisible_lpips = total_invisible_lpips / episode
    avg_success_rate = (total_success_rate - 1) / (len(results) - 1) if len(results) > 1 else 0.0

    score = (avg_invisible_psnr / 27) + (avg_invisible_ssim) + (1 - avg_invisible_lpips) + (2 * avg_success_rate)


    if combo_index is not None and isinstance(combo_index, int) and combo_index >= 0:
        try:
            combo_summary = {
                'combo_index': combo_index,
                'avg_invisible_psnr': float(avg_invisible_psnr),
                'avg_invisible_ssim': float(avg_invisible_ssim),
                'avg_invisible_lpips': float(avg_invisible_lpips),
                'avg_success_rate': float(avg_success_rate),
                'transforms': {}
            }

            for transform_type, metrics in results.items():
                combo_summary['transforms'][str(transform_type)] = {
                    'avg_l1': float(metrics['l1_error'] / episode),
                    'avg_l2': float(metrics['l2_error'] / episode),
                    'avg_def_psnr': float(metrics['defense_psnr'] / episode),
                    'avg_def_ssim': float(metrics['defense_ssim'] / episode),
                    'avg_def_lpips': float(metrics['defense_lpips'] / episode),
                    'success_rate': float(metrics['attack_success'] / episode),
                    'avg_remain_value': float(np.sum(metrics['total_remain_map'] / episode))
                }

            json_path = os.path.join(output_dir, f"combo_{combo_index}.json")


            try:
                combo_summary['text_report'] = "\n".join(output_lines)
            except Exception:
                combo_summary['text_report'] = None

            try:
                combo_summary['training_performance_file'] = file_path
            except Exception:
                combo_summary['training_performance_file'] = os.path.join(output_dir, 'training_performance.txt')

            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(combo_summary, jf, indent=2, ensure_ascii=False)
            print(f"[Info] Wrote combo JSON summary: {json_path}")
            try:
                import sys
                sys.stdout.flush()
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Failed to write combo JSON summary for combo {combo_index}: {e}")
            try:
                import sys
                sys.stdout.flush()
            except Exception:
                pass

    return score

def visualize_actions(action_history, image_indices, attr_indices, step_indices, action_names=None, train_flag=False):
    if action_names == None:
        action_names = ['PGD Space', 'Low-freq', 'Mid-freq', 'High-freq']
        
    num_actions = len(action_names) # 현재 액션의 총 개수 (4, 6, 또는 18)

    start_time = time.time()

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join("test_result_images", current_time)

    os.makedirs(save_dir, exist_ok=True)

    result_test_dir = None
    if train_flag:
        result_test_dir = "result_test"
    else:
        result_test_dir = "result_inference"

    if result_test_dir:
        for filename in os.listdir(result_test_dir):
            src_path = os.path.join(result_test_dir, filename)
            if not os.path.isfile(src_path):
                continue
            ext = os.path.splitext(filename)[1].lower()

            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']:
                dst_path = os.path.join(save_dir, filename)
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"[WARN] Failed to move {src_path} -> {dst_path}: {e}")

            elif filename in ['reward_moving_avg.txt', 'training_performance.txt']:
                dst_path = os.path.join(save_dir, filename)
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"[WARN] Failed to move {src_path} -> {dst_path}: {e}")
    else:
        print("'result_test', 'result_inference', or 'result_test_att' folder does not exist.")

    print("Creating DataFrame...")
    df = pd.DataFrame({
        'Action': action_history,
        'Image': image_indices,
        'Attribute': attr_indices,
        'Step': step_indices
    })

    unique_images = df['Image'].nunique()
    unique_attrs = df['Attribute'].nunique()
    total_steps = len(action_history)

    print(f"Data Statistics:")
    print(f"- Total images: {unique_images}")
    print(f"- Total attributes: {unique_attrs}")
    print(f"- Total steps: {total_steps}")

    is_large_dataset = unique_images > 20
    is_very_large_dataset = unique_images > 100
    is_massive_dataset = unique_images > 1000
    is_ultra_massive_dataset = unique_images > 10000

    print("1. Generating action distribution visualization...")
    plt.figure(figsize=(12, 6)) # 라벨이 길어질 수 있으므로 가로 길이를 약간 늘림
    series_action_counts = df['Action'].value_counts().sort_index()

    # 동적 액션 개수 적용
    for i in range(num_actions):
        if i not in series_action_counts.index:
            series_action_counts[i] = 0
    series_action_counts = series_action_counts.sort_index()

    plt.bar(series_action_counts.index, series_action_counts.values)
    # x축 라벨 동적 적용 및 글자 겹침 방지를 위한 회전(rotation) 추가
    plt.xticks(range(num_actions), action_names, rotation=45, ha='right', fontsize=9)
    plt.title('Action Selection Distribution Over Entire Training Process')
    plt.xlabel('Action Type')
    plt.ylabel('Selection Count')
    plt.tight_layout() # 그래프 여백 자동 조절
    plt.savefig(os.path.join(save_dir, 'action_distribution.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.pie(series_action_counts.values, labels=action_names,
            autopct='%1.1f%%', startangle=90, shadow=True, textprops={'fontsize': 8})
    plt.axis('equal')
    plt.title('Action Selection Ratio')
    plt.savefig(os.path.join(save_dir, 'action_distribution_pie.png'))
    plt.close()

    print("2. Generating action selection pattern over time visualization...")

    if is_ultra_massive_dataset:
        max_samples = 1000
        bin_size = max(1, len(action_history) // max_samples)

        steps = np.array(range(len(action_history)))
        bins = steps // bin_size

        sampled_actions = []
        sampled_steps = []

        for bin_idx in range(bins.max() + 1):
            bin_mask = bins == bin_idx
            if np.any(bin_mask):
                sampled_actions.append(np.mean(np.array(action_history)[bin_mask]))
                sampled_steps.append(np.mean(steps[bin_mask]))

        plt.figure(figsize=(15, 6))
        plt.plot(sampled_steps, sampled_actions, '-', alpha=0.8)
        plt.yticks(range(num_actions), action_names, fontsize=9)
        plt.title(f'Action Selection Pattern Over Time (Downsampled: {len(sampled_steps)} points)')
    elif is_massive_dataset:
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(action_history)), action_history, '-', alpha=0.5)
        plt.yticks(range(num_actions), action_names, fontsize=9)
        plt.title('Action Selection Pattern Over Time (Line Graph)')
    else:
        plt.figure(figsize=(15, 6))

        if is_very_large_dataset:
            plt.plot(range(len(action_history)), action_history, '.', markersize=1, alpha=0.3)
        else:
            plt.plot(range(len(action_history)), action_history, 'o', markersize=3, alpha=0.6)

        plt.yticks(range(num_actions), action_names, fontsize=9)
        plt.title('Action Selection Pattern Over Time')

    plt.xlabel('Step')
    plt.ylabel('Selected Action')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_timeline.png'))
    plt.close()

    print("3. Generating cumulative action selection visualization...")
    plt.figure(figsize=(12, 6))
    action_timeline = []
    # 동적 액션 개수 적용
    for i in range(num_actions):
        action_counts = [0]
        for a in action_history:
            if a == i:
                action_counts.append(action_counts[-1] + 1)
            else:
                action_counts.append(action_counts[-1])
        action_timeline.append(action_counts[1:])

    for i in range(num_actions):
        plt.plot(range(len(action_history)), action_timeline[i],
                label=action_names[i])

    plt.title('Cumulative Action Selections Over Training')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Selection Count')
    
    # 18개일 경우 범례가 너무 길어질 수 있으므로 밖에 배치
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8) 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_cumulative.png'))
    plt.close()

    print("4. Analyzing action selection patterns per image...")

    img_action_pivot = df.pivot_table(
        index='Image',
        columns='Action',
        aggfunc='size',
        fill_value=0
    )

    # 동적 액션 개수 적용
    for i in range(num_actions):
        if i not in img_action_pivot.columns:
            img_action_pivot[i] = 0

    img_action_pivot = img_action_pivot.reindex(sorted(img_action_pivot.columns), axis=1)
    img_action_pivot.columns = action_names

    img_action_pivot.to_csv(os.path.join(save_dir, 'image_action_counts.csv'))

    img_action_ratio = img_action_pivot.div(img_action_pivot.sum(axis=1), axis=0)
    img_action_ratio.to_csv(os.path.join(save_dir, 'image_action_ratios.csv'))

    print("5. Determining heatmap visualization strategy...")

    if is_massive_dataset:
        print("   Starting clustering analysis for large dataset...")

        sample_size = min(5000, unique_images)
        if unique_images > sample_size:
            print(f"   For memory efficiency, sampling {sample_size} images for clustering...")
            sample_images = np.random.choice(img_action_ratio.index, sample_size, replace=False)
            clustering_data = img_action_ratio.loc[sample_images]
        else:
            clustering_data = img_action_ratio

        n_clusters = min(20, max(3, sample_size // 250))

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(clustering_data.fillna(0))

            cluster_df = pd.DataFrame({
                'Image': clustering_data.index,
                'Cluster': clusters
            })

            cluster_patterns = pd.DataFrame()
            for i in range(n_clusters):
                cluster_imgs = cluster_df[cluster_df['Cluster'] == i]['Image']
                if len(cluster_imgs) > 0:
                    pattern = clustering_data.loc[cluster_imgs].mean()
                    cluster_patterns[f'Cluster_{i}'] = pattern

            plt.figure(figsize=(max(12, n_clusters * 0.8), max(6, num_actions * 0.4)))
            sns.heatmap(cluster_patterns.T, annot=True, cmap='viridis', fmt='.2f', 
                        yticklabels=action_names) # y축 라벨을 액션 이름으로 명시
            plt.title('Average Action Selection Pattern by Cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cluster_patterns.png'))
            plt.close()

            cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()

            plt.figure(figsize=(max(10, n_clusters * 0.8), 6))
            bars = plt.bar(cluster_sizes.index, cluster_sizes.values)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}', ha='center', va='bottom')

            plt.title('Number of Images per Cluster')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Images')
            plt.xticks(cluster_sizes.index)
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'cluster_sizes.png'))
            plt.close()
        except Exception as e:
             print(f"Clustering failed: {e}")


def plot_reward_trend(reward_list, window_size=25, save_path="reward_trend.png"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        episodes = np.arange(len(reward_list))
        reward_array = np.array(reward_list)


        if reward_array.size == 0:
            print(f"[WARN] plot_reward_trend: empty reward_list, skipping plot: {save_path}")
            return


        if window_size is None or window_size < 1:
            window_size = 1


        weights = np.ones(window_size) / float(window_size)
        try:
            moving_avg = np.convolve(reward_array, weights, mode='same')
        except Exception:

            moving_avg = np.array([np.mean(reward_array[max(0, i - window_size + 1):i + 1]) for i in range(len(reward_array))])


        len_eps = len(episodes)
        len_mov = len(moving_avg)

        if len_mov == len_eps:
            x_mov = episodes
            y_mov = moving_avg
            x_raw = episodes
            y_raw = reward_array
        elif len_mov < len_eps:
            x_mov = episodes[:len_mov]
            y_mov = moving_avg
            x_raw = episodes[:len_mov]
            y_raw = reward_array[:len_mov]
        else:

            x_mov = episodes
            xp = np.linspace(0, len_eps - 1, num=len_mov)
            xnew = np.arange(len_eps)
            try:
                y_mov = np.interp(xnew, xp, moving_avg)
            except Exception:

                y_mov = moving_avg[:len_eps]
            x_raw = episodes
            y_raw = reward_array

        plt.figure(figsize=(10, 5))
        plt.plot(x_mov, y_mov, label=f"Moving Average (window={window_size})", color='tab:blue')
        plt.plot(x_raw, y_raw, color='lightgray', alpha=0.4, label='Episodic Return')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Trend Over Episodes")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"[WARN] Failed to save reward trend plot {save_path}: {e}")
        plt.close()
    except Exception as e:
        print(f"[WARN] plot_reward_trend failed: {e}")
        return


def save_reward_moving_average_txt(reward_list, window_size=25, save_path="reward_moving_avg.txt"):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("==== Raw Episodic Return ====\n")
        for i, r in enumerate(reward_list):
            f.write(f"Episode {i}: {r:.4f}\n")

        f.write("\n\n==== Moving Average of Reward (Window size = {}) ====\n".format(window_size))
        f.write("=" * 50 + "\n")

        num_windows = len(reward_list) // window_size
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = reward_list[start:end]
            avg = sum(window) / len(window)
            f.write(f"Window {i+1} (Episode {start}~{end-1}): Average = {avg:.4f}\n")


        remainder = len(reward_list) % window_size
        if remainder > 0:
            start = num_windows * window_size
            end = len(reward_list)
            window = reward_list[start:end]
            avg = sum(window) / len(window)
            f.write(f"Window {num_windows+1} (Episode {start}~{end-1}): Average = {avg:.4f}\n")
