import math
from glob import glob
import mne
import numpy as np

from LSTMModel import lstm_model
from averageMetricsCalculator import calculate_average_metrics
from evaluate import evaluate_model, evaluate_model_in
from scaler import custom_scale_data
from train import train_model, train_model_in
from ts2vg import NaturalVG
import networkx as nx
from scipy.signal import welch
from sklearn.model_selection import KFold

edf_files = glob('data.edf')
all_file_paths = glob('data.edf')
healthy_file_path = [i for i in all_file_paths if 'h' in i.split('\\')[1]]
patient_file_path = [i for i in all_file_paths if 's' in i.split('\\')[1]]


def get_psd_features(file_path):
    data = mne.io.read_raw_edf(file_path, preload=True)
    data.set_eeg_reference()
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100)
    }
    epoch_duration = 1
    overlap = 0
    epochs = mne.make_fixed_length_epochs(data, duration=epoch_duration, overlap=overlap)
    epochs = epochs.get_data()
    psd_results = []
    for epoch in epochs:
        epoch_psd = []
        for channel_idx in range(len(data.ch_names)):
            channel_psd = []
            for band, (low, high) in bands.items():
                filtered_data = mne.filter.filter_data(epoch[channel_idx], sfreq=data.info['sfreq'], l_freq=low,
                                                       h_freq=high, fir_design='firwin')
                frequencies, psd = welch(filtered_data, nperseg=256, fs=data.info['sfreq'])
                channel_psd.append(psd)
            channel_psd_array = np.array(channel_psd)
            epoch_psd.append(channel_psd_array)
        epoch_psd_array = np.array(epoch_psd)
        psd_results.append(epoch_psd_array)
    psd_results_array = np.array(psd_results)
    return psd_results_array


controle_psd_array = [get_psd_features(i) for i in healthy_file_path]
patient_psd_array = [get_psd_features(i) for i in patient_file_path]


def construct_visibility_graph(psd_array):
    vgPerPatient = []
    for i in psd_array:
        vgPerEpoch = []

        for patient_idx in range(i.shape[0]):
            vgDimensions = []
            for channel_idx in range(i.shape[1]):
                feature_vector = i[patient_idx, channel_idx, :, :]
                vgPerRow = []
                for row in feature_vector:
                    ng = NaturalVG()
                    ng.build(row)
                    vgPerRow.append(ng.adjacency_matrix())
                vgDimensions.append(vgPerRow)
            vgPerEpoch.append(vgDimensions)
        vgPerPatient.append(vgPerEpoch)
    return vgPerPatient


vgPerPatient = construct_visibility_graph(controle_psd_array)
vgPerControle = construct_visibility_graph(patient_psd_array)


def extract_graph_theory_features(vg):
    perBand = []
    for i in vg:
        vgPerEpoch = []
        for epoch_idx in i:
            vgChannel = []
            for channel_idx in epoch_idx:
                vgband = []
                for band_inx in channel_idx:
                    vg = band_inx
                    degree_vec = np.sum(vg, axis=1)
                    avg_degree = np.mean(degree_vec)
                    max_degree = np.max(degree_vec)
                    adj_matrix = vg
                    g = nx.Graph(adj_matrix)
                    graph_density = nx.density(g)
                    max_cliques = list(nx.find_cliques(g))
                    if max_cliques:
                        max_clique_size = max(len(clique) for clique in max_cliques)
                    else:
                        max_clique_size = 0

                    radius = nx.radius(g)
                    diameter = nx.diameter(g)
                    independence_number = nx.graph_clique_number(g)
                    degrees = dict(g.degree())
                    degree_counts = {}
                    for degree in degrees.values():
                        if degree in degree_counts:
                            degree_counts[degree] += 1
                        else:
                            degree_counts[degree] = 1
                    total_nodes = len(g.nodes())
                    degree_distribution = {degree: count / total_nodes for degree, count in degree_counts.items()}
                    entropy = -sum(p * math.log2(p) for p in degree_distribution.values() if p > 0)
                    assortativity = nx.degree_assortativity_coefficient(g)
                    clustering_coefficient = nx.average_clustering(g)
                    global_efficiency = nx.global_efficiency(g)
                    vgband.append(np.array(
                        [avg_degree, max_degree, graph_density, max_clique_size, radius, diameter, independence_number,
                         entropy, assortativity, clustering_coefficient, global_efficiency]))
                vgChannel.append(vgband)
            vgPerEpoch.append(vgChannel)
        perBand.append(vgPerEpoch)
    return perBand


perBand = extract_graph_theory_features(vgPerPatient)
perBandPerPatient = extract_graph_theory_features(vgPerControle)

controle_epochs_labels = [len(i) * [0] for i in perBand]
patient_epochs_labels = [len(i) * [1] for i in perBandPerPatient]
data_list = perBand + perBandPerPatient
label_list = patient_epochs_labels + controle_epochs_labels
data_array = np.vstack(data_list)
lable_array = np.hstack(label_list)


def run_experiment(model_type, data_array, label_array, t=None, input_dimension=None, num_repetitions=1):
    all_accuracy_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []
    all_auc_scores = []
    all_run_fpr = []
    all_run_tpr = []

    for experiment_number in range(num_repetitions):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []
        all_fpr = []
        all_tpr = []
        for train_index, test_index in kf.split(data_array):
            train_features, test_features = data_array[train_index], data_array[test_index]
            train_labels, test_labels = label_array[train_index], label_array[test_index]
            train_features, test_features = custom_scale_data(train_features, test_features)
            accuracy = None
            precision = None
            recall = None
            f1 = None
            roc_auc = None
            fpr = None
            tpr = None
            if model_type == 'LSTM':
                inputShape = (t, input_dimension)
                model = lstm_model(inputShape)
                trained_model, history = train_model(model, train_features, train_labels)
                accuracy, precision, recall, f1, confusion, roc_auc, fpr, tpr = evaluate_model(trained_model,
                                                                                               test_features,
                                                                                               test_labels)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_scores.append(roc_auc)
            all_fpr.append(fpr)
            all_tpr.append(tpr)

        all_accuracy_scores.append(np.mean(accuracy_scores))
        all_precision_scores.append(np.mean(precision_scores))
        all_recall_scores.append(np.mean(recall_scores))
        all_f1_scores.append(np.mean(f1_scores))
        all_auc_scores.append(np.mean(auc_scores))
        all_run_fpr.extend(all_fpr)
        all_run_tpr.extend(all_tpr)

    average_accuracy, average_precision, average_recall, average_f1, average_auc = calculate_average_metrics(
        all_accuracy_scores, all_precision_scores, all_recall_scores, all_f1_scores, all_auc_scores
    )
    return all_run_fpr, all_run_tpr


batch_size, time_steps, features_1, features_2 = data_array.shape
combined_features = features_1 * features_2
reshaped_data_array = data_array.reshape(batch_size, time_steps, combined_features)
ts = data_array.shape[1]
all_run_fpr_lstm, all_run_tpr_lstm = run_experiment('LSTM', reshaped_data_array, lable_array, t=ts,
                                                    input_dimension=combined_features,
                                                    num_repetitions=10)
