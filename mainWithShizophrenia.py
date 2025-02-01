import math
from glob import glob
import mne
import numpy as np

from LSTMModel import lstm_model
from ts2vg import NaturalVG
import networkx as nx
from sklearn.model_selection import KFold

from train import train_model, train_model_in
from evaluate import evaluate_model, evaluate_model_in
from scaler import scale_data, custom_scale_data
from averageMetricsCalculator import calculate_average_metrics



def extract_file_paths():
    all_file_paths = glob('data.edf')
    healthy_file_path = [i for i in all_file_paths if 'h' in i.split('\\')[1]]
    patient_file_path = [i for i in all_file_paths if 's' in i.split('\\')[1]]
    return healthy_file_path, patient_file_path


def read_data(file_path):
    data = mne.io.read_raw_edf(file_path, preload=True)
    data.set_eeg_reference()
    data.filter(l_freq=0.5, h_freq=45)
    epochs = mne.make_fixed_length_epochs(data, duration=1, overlap=0)
    array = epochs.get_data()
    return array


healthy_file_path, patient_file_path = extract_file_paths()
controle_epochs_array = [read_data(i) for i in healthy_file_path]
patient_epochs_array = [read_data(i) for i in patient_file_path]


def ptp(data):
    return np.ptp(data, axis=-1)


def calculate_features(epochs_array):
    patientsFeatures = []
    patientFeatures = []
    for data in epochs_array:

        for i in range(data.shape[0]):
            patientFeatures.append(ptp(data[i]))
        patientsFeatures.append(patientFeatures)
        patientFeatures = []
    return patientsFeatures


patientsFeatures = calculate_features(patient_epochs_array)
controlsFeatures = calculate_features(controle_epochs_array)


def convert_list_to_array(patientsFeatures):
    patientsFeatures2 = []
    for i in range(len(patientsFeatures)):
        patientsFeatures2.append(np.vstack(patientsFeatures[i]))
    return patientsFeatures2


patientsFeatures2 = convert_list_to_array(patientsFeatures)
controlsFeatures2 = convert_list_to_array(controlsFeatures)


def reorganize_structure_to_construct_vg(features):
    num_columns = 19
    columnsListEpoControle = []
    columnsListControle = []
    for control in features:
        for i in range(num_columns):
            column = control[:, i]
            columnsListEpoControle.append(column)
        columnsListControle.append(columnsListEpoControle)
        columnsListEpoControle = []
    return columnsListControle


columnsListControle = reorganize_structure_to_construct_vg(controlsFeatures2)
columnsListPatient = reorganize_structure_to_construct_vg(patientsFeatures2)


def construct_vg(columnsList):
    num_columns = 19
    vgPatient = []
    vgsPatients = []
    for patient in columnsList:
        for i in range(num_columns):
            flFeauture = np.ravel(patient[i])
            ng = NaturalVG()
            ng.build(flFeauture)
            vgPatient.append(ng.adjacency_matrix())
        vgsPatients.append(vgPatient)
        vgPatient = []
    return vgsPatients


vgsPatients = construct_vg(columnsListPatient)
vgsControls = construct_vg(columnsListControle)


def graph_theory_features_extraction(vgs):
    patientGraphFeatureVector = []
    patientGraphsFeatureVector = []
    for patient in vgs:
        for vg in patient:
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
            patientGraphFeatureVector.append(np.array(
                [avg_degree, max_degree, graph_density, max_clique_size, radius, diameter, independence_number, entropy,
                 assortativity, clustering_coefficient, global_efficiency]))

        patientGraphsFeatureVector.append(patientGraphFeatureVector)
        patientGraphFeatureVector = []
    return patientGraphsFeatureVector


patientGraphsFeatureVector = graph_theory_features_extraction(vgsPatients)
controlsGraphsFeatureVector = graph_theory_features_extraction(vgsControls)

controle_epochs_labels = [len(i) * [0] for i in patientGraphsFeatureVector]
patient_epochs_labels = [len(i) * [1] for i in controlsGraphsFeatureVector]

data_list = patientGraphsFeatureVector + controlsGraphsFeatureVector
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

            if model_type == 'LSTM':
                train_features, test_features = custom_scale_data(train_features, test_features)
            else:
                train_features, test_features = scale_data(train_features, test_features)

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



data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[1])
input_dimension = data_array.shape[2]
t = 1
all_run_fpr_lstm, all_run_tpr_lstm = run_experiment('LSTM', data_array, lable_array, t, input_dimension,
                                                    num_repetitions=10)
