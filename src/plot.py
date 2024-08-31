import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Net
from train import test_data


def test(net, device, test_loader, top_k=5):
    total_policy_loss = 0.0
    total_value_loss = 0.0

    correct_top1 = 0
    correct_topk_weighted = 0.0
    total_topk_weighted = 0.0

    total_samples = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            boards, true_policies, true_values = data
            boards = boards.to(device)
            true_policies = true_policies.to(device)
            true_values = true_values.to(device)

            # Predict with the network
            y_policies, y_values = net(boards)

            # Calculate policy loss (Cross-Entropy)
            policy_loss = -torch.sum(true_policies *
                                     torch.log(1e-6 + y_policies))
            total_policy_loss += policy_loss.item()

            # Calculate value loss (MSE)
            value_loss = F.mse_loss(
                y_values.squeeze(), true_values, reduction='sum')
            total_value_loss += value_loss.item()

            # Calculate top-1 accuracy
            top1_pred = torch.argmax(y_policies, dim=1)
            correct_top1 += (top1_pred ==
                             torch.argmax(true_policies, dim=1)).sum().item()

            # Calculate top-k weighted accuracy
            topk_pred = torch.topk(y_policies, top_k, dim=1).indices
            topk_probs = torch.gather(true_policies, 1, topk_pred)

            correct_topk_weighted += topk_probs.sum().item()
            total_topk_weighted += true_policies.sum().item()

            total_samples += boards.size(0)

    # Calculate averages
    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples
    top1_accuracy = correct_top1 / total_samples
    topk_weighted_accuracy = correct_topk_weighted / total_topk_weighted

    print(f"Policy Loss: {avg_policy_loss:.4f}")
    print(f"Value Loss: {avg_value_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-{top_k} Weighted Accuracy: {topk_weighted_accuracy:.4f}")

    return avg_policy_loss, avg_value_loss, top1_accuracy, topk_weighted_accuracy


def plot_results(results, top_k=5):
    plt.figure(figsize=(14, 10))

    for res_block, epochs, metrics in results:
        epochs = list(epochs)
        policy_losses = [m[0] for m in metrics]
        value_losses = [m[1] for m in metrics]
        top1_accuracies = [m[2] for m in metrics]
        topk_weighted_accuracies = [m[3] for m in metrics]

        plt.subplot(4, 1, 1)
        plt.plot(epochs, policy_losses, label=f'res{res_block}')
        plt.xlabel('epochs')
        plt.ylabel('policy loss')
        plt.title('policy loss over time')

        plt.subplot(4, 1, 2)
        plt.plot(epochs, value_losses, label=f'res{res_block}')
        plt.xlabel('epochs')
        plt.ylabel('value loss')
        plt.title('value loss over time')

        plt.subplot(4, 1, 3)
        plt.plot(epochs, top1_accuracies, label=f'res{res_block}')
        plt.xlabel('epochs')
        plt.ylabel('top-1 accuracy')
        plt.title('top-1 accuracy over time')

        plt.subplot(4, 1, 4)
        plt.plot(epochs, topk_weighted_accuracies, label=f'res{res_block}')
        plt.xlabel('epochs')
        plt.ylabel(f'top-{top_k} weighted accuracy')
        plt.title(f'top-{top_k} weighted accuracy over time')

    plt.subplot(4, 1, 1)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    models = [
        (5, [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 283]),
        (20, [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300])
    ]
    results = []

    for res_blocks, epochs in models:
        net = Net(res_blocks).to(device)
        metrics = []
        for epoch in epochs:
            path = f'./models/resd/res{res_blocks}d_{epoch}.pth'

            net.load_state_dict(torch.load(path))
            net.eval()

            print(f'TESTING RES{res_blocks}_{epoch}')
            metrics.append(test(net, device, test_loader))

        results.append((res_blocks, epochs, metrics))

    plot_results(results)
