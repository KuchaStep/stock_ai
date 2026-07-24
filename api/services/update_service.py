import subprocess


def update_data():

    subprocess.run(["python", "scripts/collect_all_stocks_v2.py"])

    subprocess.run(["python", "scripts/update_latest.py"])

    subprocess.run(["python", "scripts/create_features_v4.py"])

    subprocess.run(["python", "scripts/create_target_v4.py"])
    
    return {
        "status": "success"
    }