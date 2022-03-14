DATA_BASE_PATH = {
    'tuini15-cg05-cg-in-tum-de': '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/converted_normal_anomaly',
    'gpusrv01-cg-in-tum-de': '/home/hoehlein/data/1000_member_ensemble/normalized_anomalies/single_member'
}


def get_data_base_path():
    import socket
    host_name = socket.gethostname()
    return DATA_BASE_PATH[host_name]
