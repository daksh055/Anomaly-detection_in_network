import pandas as pd

def load_kdd_data():
    """Load and preprocess KDD Cup 1999 data"""
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label"
    ]
    
    df = pd.read_csv('kddcup.data_10_percent.gz', header=None, names=columns)
    
    # Convert labels
    df['is_attack'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # Select only numerical features for the model
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_features = numerical_features.drop('is_attack', errors='ignore')
    
    return df[numerical_features], df['is_attack']