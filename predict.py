import torch
import esm
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():

    # 1. Load ab ag seq
    print('Task start time:',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (CPU execution is forbidden).")

    device = torch.device("cuda:0")
    print("Running on the GPU")

    agseq, abseq = loadseq()
    dataset = [('agseq', agseq.upper()), ('abseq', abseq.upper())]

    # 2. Represent sequence using esm-2
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    print("Sequence characterization begins", end="")
    representations_list = []
    for i in range(len(dataset)):
        data = [dataset[i]]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens,
                            repr_layers=[33],
                            return_contacts=True)
        token_representations = results["representations"][33]
        representations_list.append(token_representations[0][1:-1])
        print('.', end="")
    print('\tDone')

    # 3. predict
    model1 = torch.load('./model/model_ag.pth').to(device)
    model2 = torch.load('./model/model_ab.pth').to(device)
    model1.eval()
    model2.eval()

    # apply model
    input_sequence1 = representations_list[0]  # ag feature matrix (already on GPU)
    input_sequence2 = representations_list[1]  # ab feature matrix (already on GPU)
    print("Model prediction begins", end="")
    output1 = model1(input_sequence1)
    output2 = model2(input_sequence2)

    result = torch.matmul(output1, output2.transpose(0, 1))
    result = torch.sigmoid(result)
    print("\tDone")
    print(result.shape)

    # set threshold
    threshold = 0.5

    # Save/visualize requires CPU numpy
    result_cpu = result.detach().float().cpu()

    import pandas as pd
    import os
    os.makedirs("./results", exist_ok=True)

    res_df = pd.DataFrame(result_cpu.numpy())
    res_df.to_csv("./results/prediction_matrix_scores.csv", index=False, header=False)
    print(">>> 所有的预测概率矩阵已保存至 ./results/prediction_matrix_scores.csv !!!")

    result_binary = np.where(result_cpu.numpy() >= threshold, 1, 0)

    print('threshold: ', threshold)

    # Find the coordinate with value 1
    indices = np.where(result_binary == 1)
    coordinates = list(zip(indices[0] + 1, indices[1] + 1))

    # print sites
    print('---------------------------------------------------------------')
    print("predict_sites: ", coordinates)

    # print epitope and paratope
    predict_epitope = sorted(set([x for x, y in coordinates]))
    predict_paratope = sorted(set([y for x, y in coordinates]))
    print('---------------------------------------------------------------')
    print('predict_epitope:  ', predict_epitope)
    print('predict_paratope: ', predict_paratope)

    # 4. visualization
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(result_cpu.numpy(), cmap="viridis", cbar=True,
                xticklabels=False, yticklabels=False)
    plt.title("Epitope-Paratope Interaction Probability Matrix")
    plt.xlabel("Antibody Residues")
    plt.ylabel("Antigen Residues")
    plt.tight_layout()
    plt.savefig("./results/prediction_heatmap.png", dpi=300)
    print(">>> 交互概率矩阵可视化已生成: ./results/prediction_heatmap.png")

    print('Task end time::',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def loadseq():

    parser = argparse.ArgumentParser(description='please set the antigen and antibody sequences.')
    parser.add_argument(
        '--agseq',
        default=
        'MQIPQAPWPVVWAVLQLGWRPGWFLDSPDRPWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDCRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAEVPTAHPSPSPRPAGQFQTLVVGVVGGLLGSLVLLVWVLAVICSRAARGTIGARRTGQPLKEDPSAVPVFSVDYGELDFQWREKTPEPPVPCVPEQTEYATIVFPSGMGTSSPARRGSADGPRSAQPLRPEDGHCSWPL',
        help='Description of abseq parameter')
    parser.add_argument(
        '--abseq',
        default=
        'evqllesggvlvqpggslrlscaasgftfsnfgmtwvrqapgkglewvsgisgggrdtyfadsvkgrftisrdnskntlylqmnslkgedtavyycvkwgniyfdywgqgtlvtvssastkgpsvfplapcsrstsestaalgclvkdyfpepvtvswnsgaltsgvhtfpavlqssglyslssvvtvpssslgtktytcnvdhkpsntkvdkrveskygppcppcpapeflggpsvflfppkpkdtlmisrtpevtcvvvdvsqedpevqfnwyvdgvevhnaktkpreeqfnstyrvvsvltvlhqdwlngkeykckvsnkglpssiektiskakgqprepqvytlppsqeemtknqvsltclvkgfypsdiavewesngqpennykttppvldsdgsfflysrltvdksrwqegnvfscsvmhealhnhytqkslslslgk',
        help='Description of agseq parameter')

    # Parse command line arguments
    args = parser.parse_args()

    # Access the value of the parameter
    agseq_value = args.agseq
    abseq_value = args.abseq

    return agseq_value, abseq_value


if __name__ == "__main__":
    main()
