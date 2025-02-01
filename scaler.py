from sklearn.preprocessing import StandardScaler


def scale_data(train_features, test_features):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    return train_features_scaled, test_features_scaled


def custom_scale_data(train_features, test_features):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(
        train_features.shape)
    test_features_scaled = scaler.transform(test_features.reshape(-1, test_features.shape[-1])).reshape(
        test_features.shape)
    return train_features_scaled, test_features_scaled

